# This file is for the streamlit frontend

# All imports
import os
import sys
import traceback
import contextlib
from io import StringIO
from dotenv import load_dotenv
import streamlit as st
import statsmodels.api as sm
from streamlit_feedback import streamlit_feedback
from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from retrievers import *
from agents import *
import plotly as px
import matplotlib.pyplot as plt

load_dotenv()

def reset_everything():
    st.cache_data.clear()

# stores std output generated on the console to be shown to the user on streamlit app
@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old
    
    
# Markdown changes for the streamlit app
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #ff8080;
        color: #ffffff;
        title-color:#ffffff
    }
</style>
<style>
  [data-testid=stSidebar] h1 {
    color: #ffffff

   }
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .css-1l1u5u8 code {
        color: black; /* Change this to your desired color */
        background-color: #f5f5f5; /* Optional: change background color if needed */
        padding: 2px 4px;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

AGENT_NAMES =  [data_viz_agent,sk_learn_agent,statistical_analytics_agent,preprocessing_agent]

# initializes some variables in the streamlit session state
# messages used for storing query and agent responses
if "messages" not in st.session_state:
    st.session_state.messages = []
# thumbs used to store user feedback
if "thumbs" not in st.session_state:
    st.session_state.thumbs = ''
#stores df
if "df" not in st.session_state:
    st.session_state.df = None
#stores short-term memory
if "st_memory" not in st.session_state:
    st.session_state.st_memory = []
if "configured_lm" not in st.session_state:
    st.session_state.configured_lm = None


def configure_lm(model="gpt-4o-mini", api_key=os.environ['OPENAI_API_KEY']):
    """Configures the OpenAI model and returns the model object"""
    dspy.configure(
        lm = dspy.OpenAI(model='gpt-4o-mini',
            api_key=os.environ['OPENAI_API_KEY'], 
            max_tokens=16384)
     )
    Settings.embed_model = OpenAIEmbedding(api_key=os.environ["OPENAI_API_KEY"])


# Imports images
st.image('./images/Auto-analysts icon small.png', width=70)
st.title("Auto-Analyst")
    

# asthetic features for streamlit app
st.logo('./images/Auto-analysts icon small.png')
st.sidebar.title(":white[Auto-Analyst] ")
st.sidebar.text("Have all your Data Science ")
st.sidebar.text("Analysis Done!")

def upload_file():
    uploaded_file = st.file_uploader("Upload your file here...", on_change=reset_everything())
    st.write("You can upload your own data or use sample data by clicking the button below")
    # Creates a button to use the sample data
    sample_data = st.button("Use Sample Data")
    if sample_data:
        uploaded_file = "Housing.csv"
    return uploaded_file, sample_data
        
# Displays the instructions for the user
st.markdown(instructions)

retrievers = {}
# Initializes the uploaded_df or sample df into a dataframe & also caches that data for performance
@st.cache_data
def initialize_data(uploaded_file, button_pressed=False):
    if button_pressed==False:
        uploaded_df = pd.read_csv(uploaded_file, parse_dates=True)
    else:
        uploaded_df = pd.read_csv("Housing.csv")
        st.write("LOADED")
    return uploaded_df

# Initializes the planner based system
@st.cache_resource
def intialize_agent():
    return auto_analyst(agents=AGENT_NAMES,retrievers=retrievers)

# Intializes the independent components
@st.cache_resource
def initial_agent_ind():
    return auto_analyst_ind(agents=AGENT_NAMES,retrievers=retrievers)

# Initializes the two retrievers one for data, the other for styling to be used by visualization agent
@st.cache_data(hash_funcs={StringIO: StringIO.getvalue})
def initiatlize_retrievers(_styling_instructions, _doc):
    style_index =  VectorStoreIndex.from_documents([Document(text=x) for x in _styling_instructions])
    retrievers['style_index'] = style_index
    retrievers['dataframe_index'] =  VectorStoreIndex.from_documents([Document(text=x) for x in _doc])

    return retrievers
    
# A simple function to save the output 
def save():
    filename = 'logs.txt'
    outfile = open(filename, 'a')
    
    outfile.writelines([str(i)+'\n' for i in st.session_state.messages])
    outfile.close()

# Defines how the chat system works
def run_chat():
    # Defines a variable df (agent code often refers to the dataframe as that)
    if 'df' in st.session_state:
        df = st.session_state['df']
        if df is not None:
            st.write(df.head(5))
            if "show_placeholder" not in st.session_state:
                st.session_state.show_placeholder = True
        else:
            st.error("No data uploaded yet, please upload a file or use sample data")
   
    # Placeholder text to display above the chat box
    placeholder_text = "Welcome to Auto-Analyst, How can I help you? You can use @agent_name to call a specific agent or let the planner route the query!"

    # Display the placeholder text above the chat box
    if "show_placeholder" in st.session_state and st.session_state.show_placeholder:
        st.markdown(f"**{placeholder_text}**")

    # User input taken here    
    user_input = st.chat_input("What are the summary statistics of the data?")

    # Once the user enters a query, hide the placeholder text
    if user_input:
        st.session_state.show_placeholder = False

    # If user has given input or query
    if user_input:
        # this chunk displays previous interactions
        if st.session_state.messages!=[]:
            for m in st.session_state.messages:
                if '-------------------------' not in m:
                    st.write(m.replace('#','######'))

        st.session_state.messages.append('\n------------------------------------------------NEW QUERY------------------------------------------------\n')
        st.session_state.messages.append(f"User: {user_input}")
        
        # All the agents the user mentioned by name to be stored in this list
        specified_agents = []
        # Checks for each agent if it is mentioned in the query
        for a in AGENT_NAMES: 
            if a.__pydantic_core_schema__['schema']['model_name'] in user_input.lower():
                specified_agents.insert(0,a.__pydantic_core_schema__['schema']['model_name'])

    # This is triggered when user did not mention any of the agents in the query; a planner based routing
        if specified_agents==[]:
            # Generate response in a chat message object
            with st.chat_message("Auto-Anlyst Bot",avatar="🚀"):
                st.write("Responding to "+ user_input)
                # sends the query to the chat system
                output=st.session_state['agent_system_chat'](query=user_input)
                #only executes output from the code combiner agent
                execution = output['code_combiner_agent'].refined_complete_code.split('```')[1].replace('#','####').replace('python','')
                st.markdown(output['code_combiner_agent'].refined_complete_code)
                
                # Tries to execute the code and display the output generated from the console
                try:
                    with stdoutIO() as s:
                        exec(execution)
                    st.write(s.getvalue().replace('#','########'))
                # If code generates an error (testing code fixing agent will be added here)
                except:
                    e = traceback.format_exc()
                    st.markdown("The code is giving an error on excution "+str(e)[:1500])
                    st.write("Please help the code fix agent with human understanding")
                    user_given_context = st.text_input("Help give additional context to guide the agent to fix the code", key='user_given_context')
                    st.session_state.messages.append(user_given_context)

    # this is if the specified_agent list is not empty, send to individual mentioned agents
        else:
            for spec_agent in specified_agents:
                with st.chat_message(spec_agent+" Bot",avatar="🚀"):
                    st.markdown("Responding to "+ user_input)
                    # only sends to the specified agents 
                    output=st.session_state['agent_system_chat_ind'](query=user_input, specified_agent=spec_agent)

                    # Fail safe sometimes code output not structured correctly
                    if len(output[spec_agent].code.split('```'))>1:
                        execution = output[spec_agent].code.split('```')[1].replace('#','####').replace('python','').replace('fig.show()','st.plotly_chart(fig)')
                    else:
                        execution = output[spec_agent].code.split('```')[0].replace('#','####').replace('python','').replace('fig.show()','st.plotly_chart(fig)')
                    # does the code execution and displays it to the user
                    try:
                        with stdoutIO() as s:
                            exec(execution)
                        st.write(s.getvalue().replace('#','########'))                        
                # If code generates an error (testing code fixing agent will be added here)
                    except:
                        e = traceback.format_exc()
                        st.markdown("The code is giving an error on excution "+str(e)[:1500])
                        st.write("Please help the code fix agent with human understanding")
                        user_given_context = st.text_input("Help give additional context to guide the agent to fix the code", key='user_given_context')
                        st.session_state.messages.append(user_given_context)
        
        # Simple feedback form to capture the user's feedback on the answers
        with st.form('form'):
            streamlit_feedback(feedback_type="thumbs", optional_text_label="Do you like the response?", align="flex-start")

            st.session_state.messages.append('\n---------------------------------------------------------------------------------------------------------\n')
            st.form_submit_button('Save feedback',on_click=save())

# Function to handle file upload or sample data selection
def handle_file_upload_or_sample(sample_data, uploaded_file):
    if sample_data:
        desc = "Housing Dataset"
        doc = [str(make_data(st.session_state['df'], desc))]
    else:
        desc = st.text_input("Write a description for the uploaded dataset")
        doc = ['']
        if st.button("Start The Analysis"):
            dict_ = make_data(st.session_state['df'], desc)
            doc = [str(dict_)]
    return doc

# Function to initialize retrievers
def initialize_retrievers_and_agents(doc):
    if doc[0] != '':
        retrievers = initiatlize_retrievers(styling_instructions, doc)
        st.success('Document Uploaded Successfully!')
        st.session_state['agent_system_chat'] = intialize_agent()
        st.session_state['agent_system_chat_ind'] = initial_agent_ind()
        st.write("Begin")

# Function to save user feedback
def save_feedback():
    if st.session_state['thumbs'] != '':
        filename = 'output2.txt'
        with open(filename, 'a', encoding="utf-8") as outfile:
            outfile.write(str(st.session_state.thumbs) + '\n')
            outfile.write('\n------------------------------------------------END QUERY------------------------------------------------\n')
        st.session_state['thumbs'] = ''
        st.write("Saved your Feedback")

# Function to manage short-term memory
def manage_short_term_memory():
    if len(st.session_state.st_memory) > 10:
        st.session_state.st_memory = st.session_state.st_memory[:10]

# Main function that runs the app
def run_app():
    if st.session_state.configured_lm is None:
        configure_lm()
    uploaded_file, sample_data = upload_file()
    if uploaded_file or sample_data:
        # Initialize the dataframe
        st.session_state['df'] = initialize_data(uploaded_file)
        
        # Handle file upload or sample data
        doc = handle_file_upload_or_sample(sample_data, uploaded_file)
        
        # Initialize retrievers and agents
        initialize_retrievers_and_agents(doc)
    
    # Save user feedback
    save_feedback()
    
    # Run the chat interface
    run_chat()
    
    # Manage short-term memory
    manage_short_term_memory()

# Run the app
run_app()