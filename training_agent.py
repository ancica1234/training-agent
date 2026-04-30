from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    student_name: str
    step_count: int
    human_approved: bool

MOCK_SCHEDULES = {'25-4': {'className': '25-4', 'startDate': '01-01-25', 'events': [
    {'eventCode': '0100', 'scheduledDate': '01-01-25', 'description': 'Introduction Flight'},
    {'eventCode': '0101', 'scheduledDate': '08-01-25', 'description': 'Basic Maneuvers'},
    {'eventCode': '0102', 'scheduledDate': '15-01-25', 'description': 'Navigation'},
    {'eventCode': '0103', 'scheduledDate': '22-01-25', 'description': 'Formation Flying'},
    {'eventCode': '0104', 'scheduledDate': '29-01-25', 'description': 'Night Operations'},
]}}

MOCK_STUDENT_HISTORY = {
    'John Smith': {'className': '25-4', 'workdaysBehind': 5,
        'incompleteEvents': ['0102','0103','0104'],
        'completedEvents': [
            {'eventCode': '0100', 'completedDate': '01-01-25', 'status': 'Satisfactory'},
            {'eventCode': '0101', 'completedDate': '09-01-25', 'status': 'Satisfactory'},
        ]},
    'Jane Doe': {'className': '25-4', 'workdaysBehind': 0,
        'incompleteEvents': ['0104'],
        'completedEvents': [
            {'eventCode': '0100', 'completedDate': '01-01-25', 'status': 'Satisfactory'},
            {'eventCode': '0101', 'completedDate': '08-01-25', 'status': 'Satisfactory'},
            {'eventCode': '0102', 'completedDate': '15-01-25', 'status': 'Satisfactory'},
            {'eventCode': '0103', 'completedDate': '22-01-25', 'status': 'Satisfactory'},
        ]},
}

MOCK_REMEDIATION = {
    'makeup': 'Schedule makeup sessions on weekends to cover missed events 0102 and 0103',
    'tutoring': 'Pair with a high-performing student for guided practice on Navigation',
    'simulator': 'Add 4 extra simulator hours to build confidence before live events',
}

@tool
def get_class_schedule(class_name: str) -> str:
    """Fetch the training schedule for a given class including all events and dates."""
    s = MOCK_SCHEDULES.get(class_name)
    if not s: return 'No schedule found for class ' + class_name
    out = 'Class ' + class_name + ' started ' + s['startDate']
    for e in s['events']:
        out += ' | ' + e['eventCode'] + ' ' + e['description'] + ' ' + e['scheduledDate']
    return out

@tool
def get_student_history(student_name: str) -> str:
    """Fetch student event completion history including completed, incomplete events and workdays behind."""
    h = MOCK_STUDENT_HISTORY.get(student_name)
    if not h: return 'No history found for ' + student_name
    out = 'Student: ' + student_name
    out += ' | Completed: ' + str(len(h['completedEvents']))
    out += ' | Incomplete: ' + str(len(h['incompleteEvents']))
    out += ' | Workdays Behind: ' + str(h['workdaysBehind'])
    for e in h['completedEvents']:
        out += ' || DONE ' + e['eventCode'] + ' ' + e['status']
    for e in h['incompleteEvents']:
        out += ' || PENDING ' + e
    return out

@tool
def get_remediation_options(student_name: str) -> str:
    """Get remediation options for a student who is behind schedule."""
    h = MOCK_STUDENT_HISTORY.get(student_name)
    if not h or h['workdaysBehind'] == 0: return student_name + ' is on track.'
    out = 'Remediation options for ' + student_name + ':'
    for i, (k, v) in enumerate(MOCK_REMEDIATION.items()):
        out += ' | Option ' + str(i+1) + ': ' + v
    return out

tools = [get_class_schedule, get_student_history, get_remediation_options]
tool_node = ToolNode(tools)
llm = None
llm_with_tools = None

def agent_node(state: AgentState):
    global llm, llm_with_tools
    if llm is None:
        llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0)
        llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state['messages'])
    return {'messages': [response], 'step_count': state.get('step_count', 0) + 1}

def human_approval_node(state: AgentState):
    print('='*60)
    print('HUMAN APPROVAL REQUIRED')
    print('='*60)
    last_ai = next((m for m in reversed(state['messages']) if isinstance(m, AIMessage)), None)
    if last_ai: print(last_ai.content)
    while True:
        choice = input('Approve this plan? (yes/no): ').strip().lower()
        if choice in ['yes', 'no']: break
        print('Please type yes or no')
    approved = choice == 'yes'
    msg = 'APPROVED - Finalize next steps.' if approved else 'REJECTED - Please revise.'
    return {'messages': [HumanMessage(content=msg)], 'human_approved': approved}

def should_continue(state: AgentState):
    if state.get('step_count', 0) >= 10: return END
    last = state['messages'][-1]
    if hasattr(last, 'tool_calls') and last.tool_calls: return 'tools'
    if not state.get('human_approved') and isinstance(last, AIMessage):
        if any(w in last.content.lower() for w in ['remediation','behind','option','recommend']):
            return 'human_approval'
    return END

def after_approval(state: AgentState): return 'agent'

workflow = StateGraph(AgentState)
workflow.add_node('agent', agent_node)
workflow.add_node('tools', tool_node)
workflow.add_node('human_approval', human_approval_node)
workflow.add_edge(START, 'agent')
workflow.add_conditional_edges('agent', should_continue)
workflow.add_edge('tools', 'agent')
workflow.add_conditional_edges('human_approval', after_approval)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

def run_agent(student_name: str, class_name: str):
    print('='*60)
    print('Training Progress Agent')
    print('Student: ' + student_name + '  |  Class: ' + class_name)
    print('='*60)
    config = {'configurable': {'thread_id': student_name + '_' + class_name}}
    prompt = 'Evaluate training progress for ' + student_name + ' in class ' + class_name + '. Fetch the schedule, then history, evaluate if on track or behind, and if behind get remediation options and recommend best approach.'
    inputs = {'messages': [HumanMessage(content=prompt)], 'student_name': student_name, 'step_count': 0, 'human_approved': False}
    for chunk in app.stream(inputs, config=config, stream_mode='values'):
        chunk['messages'][-1].pretty_print()
        print()

if __name__ == '__main__':
    print('Student Training Progress Agent')
    print('1. John Smith - 5 workdays BEHIND')
    print('2. Jane Doe   - ON TRACK')
    choice = input('Pick a student (1 or 2): ').strip()
    if choice == '1': run_agent('John Smith', '25-4')
    elif choice == '2': run_agent('Jane Doe', '25-4')
    else: run_agent('John Smith', '25-4')
