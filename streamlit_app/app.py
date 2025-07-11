import streamlit as st

st.title('Secops Copilot')

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

user_input = st.text_input('You:')
if st.button('Send') and user_input:
    st.session_state['messages'].append({'role': 'user', 'content': user_input})
    # Placeholder for bot response
    st.session_state['messages'].append({'role': 'bot', 'content': 'This is a placeholder response.'})

for msg in st.session_state['messages']:
    st.write(f"**{msg['role'].capitalize()}:** {msg['content']}") 