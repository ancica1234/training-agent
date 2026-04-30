import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage
from training_agent import app, MOCK_STUDENT_HISTORY

st.set_page_config(page_title='Training Progress Agent', layout='wide')
st.title('Training Progress Agent')
st.markdown('AI-powered student training evaluator with human approval')
st.divider()

# Sidebar
st.sidebar.header('Student Selection')
student = st.sidebar.selectbox('Select Student', ['John Smith', 'Jane Doe'])
class_name = st.sidebar.text_input('Class Name', value='25-4')
api_key = st.sidebar.text_input('Groq API Key', type='password')
run_btn = st.sidebar.button('Run Agent', type='primary')
st.sidebar.divider()
st.sidebar.subheader('Student Status')
for name, data in MOCK_STUDENT_HISTORY.items():
    behind = data['workdaysBehind']
    icon = 'behind' if behind > 0 else 'on track'
    st.sidebar.write(name + ': ' + icon + ' (' + str(behind) + ' days behind)')

# Main area
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader('Agent Conversation')
    chat_box = st.container()
with col2:
    st.subheader('Agent Status')
    status_box = st.empty()

# Session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'approved' not in st.session_state:
    st.session_state.approved = None
if 'waiting_approval' not in st.session_state:
    st.session_state.waiting_approval = False

# Display existing messages
with chat_box:
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            st.chat_message('user').write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message('assistant').write(msg.content)

# Run agent when button clicked
if run_btn:
    if not api_key:
        st.error('Please enter your Groq API key in the sidebar')
    else:
        os.environ['GROQ_API_KEY'] = api_key
        st.session_state.messages = []
        st.session_state.waiting_approval = False
        st.session_state.approved = None
        config = {'configurable': {'thread_id': student + '_' + class_name}}
        prompt = 'Evaluate training progress for ' + student + ' in class ' + class_name + '. Fetch schedule, then history, evaluate if on track or behind, and if behind get remediation options and recommend best approach.'
        inputs = {'messages': [HumanMessage(content=prompt)], 'student_name': student, 'step_count': 0, 'human_approved': False}

        with st.spinner('Agent is working...'):
            for chunk in app.stream(inputs, config=config, stream_mode='values'):
                last = chunk['messages'][-1]
                if last not in st.session_state.messages:
                    st.session_state.messages.append(last)
                    if isinstance(last, AIMessage):
                        with chat_box:
                            st.chat_message('assistant').write(last.content)
                    keywords = ['remediation', 'behind', 'option', 'recommend']
                    if isinstance(last, AIMessage) and any(w in last.content.lower() for w in keywords):
                        st.session_state.waiting_approval = True
                        break

# Human approval section
if st.session_state.waiting_approval:
    st.divider()
    st.subheader('Supervisor Approval Required')
    st.warning('The agent has generated a remediation plan. Please review and approve or reject.')
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button('Approve Plan', type='primary'):
            st.session_state.approved = True
            st.session_state.waiting_approval = False
            st.rerun()
    with col_b:
        if st.button('Reject Plan'):
            st.session_state.approved = False
            st.session_state.waiting_approval = False
            st.rerun()

# Continue after approval
if st.session_state.approved is not None:
    config = {'configurable': {'thread_id': student + '_' + class_name}}
    if st.session_state.approved:
        msg = 'APPROVED - Please finalize and summarize next steps for the student.'
        st.success('Plan Approved! Agent is finalizing...')
    else:
        msg = 'REJECTED - Please revise and suggest a different approach.'
        st.error('Plan Rejected. Agent is revising...')
    follow_up = {'messages': [HumanMessage(content=msg)], 'student_name': student, 'step_count': 0, 'human_approved': st.session_state.approved}
    with st.spinner('Agent is finalizing...'):
        for chunk in app.stream(follow_up, config=config, stream_mode='values'):
            last = chunk['messages'][-1]
            if isinstance(last, AIMessage):
                st.chat_message('assistant').write(last.content)
    st.session_state.approved = None

