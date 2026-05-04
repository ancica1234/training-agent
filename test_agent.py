import pytest
from training_agent import get_class_schedule, get_student_history, get_remediation_options, should_continue, MOCK_STUDENT_HISTORY, MOCK_SCHEDULES
from langchain_core.messages import AIMessage, HumanMessage

# ============================================================
# TOOL TESTS
# ============================================================

def test_get_class_schedule_returns_data():
    result = get_class_schedule.invoke({'class_name': '25-4'})
    assert '25-4' in result
    assert '0100' in result
    assert 'Introduction Flight' in result

def test_get_class_schedule_invalid_class():
    result = get_class_schedule.invoke({'class_name': 'INVALID'})
    assert 'No schedule found' in result

def test_schedule_has_5_events():
    events = MOCK_SCHEDULES['25-4']['events']
    assert len(events) == 5

def test_get_student_history_john_smith():
    result = get_student_history.invoke({'student_name': 'John Smith'})
    assert 'John Smith' in result
    assert '5' in result
    assert '0102' in result

def test_john_smith_is_5_days_behind():
    assert MOCK_STUDENT_HISTORY['John Smith']['workdaysBehind'] == 5

def test_jane_doe_is_on_track():
    assert MOCK_STUDENT_HISTORY['Jane Doe']['workdaysBehind'] == 0

def test_get_student_history_invalid_student():
    result = get_student_history.invoke({'student_name': 'Unknown Person'})
    assert 'No history found' in result

def test_remediation_for_john_smith():
    result = get_remediation_options.invoke({'student_name': 'John Smith'})
    assert 'Option 1' in result
    assert 'Option 2' in result
    assert 'Option 3' in result

def test_no_remediation_for_jane_doe():
    result = get_remediation_options.invoke({'student_name': 'Jane Doe'})
    assert 'on track' in result

def test_remediation_for_invalid_student():
    result = get_remediation_options.invoke({'student_name': 'Unknown'})
    assert 'on track' in result

# ============================================================
# ROUTING TESTS
# ============================================================

def test_routes_to_end_when_no_tool_calls():
    msg = AIMessage(content='Student is on track.')
    state = {'messages': [msg], 'step_count': 1, 'human_approved': False}
    result = should_continue(state)
    from langgraph.graph import END
    assert result == END

def test_routes_to_human_approval_when_behind_keyword():
    msg = AIMessage(content='John Smith is behind and needs remediation options.')
    state = {'messages': [msg], 'step_count': 1, 'human_approved': False}
    result = should_continue(state)
    assert result == 'human_approval'

def test_routes_to_end_when_already_approved():
    msg = AIMessage(content='John Smith is behind and needs remediation options.')
    state = {'messages': [msg], 'step_count': 1, 'human_approved': True}
    result = should_continue(state)
    from langgraph.graph import END
    assert result == END

def test_safety_limit_stops_agent():
    msg = AIMessage(content='some response')
    state = {'messages': [msg], 'step_count': 10, 'human_approved': False}
    result = should_continue(state)
    from langgraph.graph import END
    assert result == END

# ============================================================
# ANTI-HALLUCINATION TESTS
# ============================================================

def test_john_smith_has_3_incomplete_events():
    incomplete = MOCK_STUDENT_HISTORY['John Smith']['incompleteEvents']
    assert len(incomplete) == 3
    assert '0102' in incomplete
    assert '0103' in incomplete
    assert '0104' in incomplete

def test_jane_doe_has_1_incomplete_event():
    incomplete = MOCK_STUDENT_HISTORY['Jane Doe']['incompleteEvents']
    assert len(incomplete) == 1
    assert '0104' in incomplete

def test_tool_output_matches_mock_data():
    result = get_student_history.invoke({'student_name': 'John Smith'})
    assert 'DONE 0100' in result
    assert 'DONE 0101' in result
    assert 'PENDING 0102' in result

