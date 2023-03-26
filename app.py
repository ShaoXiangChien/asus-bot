import streamlit as st
import streamlit_authenticator as stauth
from component import message
import requests
import openai
from streamlit_elements import mui, elements, html
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
import numpy as np
import azure.cognitiveservices.speech as speechsdk
import uuid

openai.api_type = 'azure'
openai.api_version = "2022-12-01"
openai.api_key = st.secrets['api']['openai_key']
openai.api_base = st.secrets['api']['openai_base']
system_message_template = "<|im_start|>system\n{}\n<|im_end|>"


st.set_page_config(
    page_title="OpenAI Chat - Demo",
)

authenticator = stauth.Authenticate(
    {
        "usernames": {"mtcuser": st.secrets['mtcuser']}
    },
    st.secrets['cookie']['name'],
    st.secrets['cookie']['key'],
    st.secrets['cookie']['expiry_days'],
)

top_cols = st.columns([3, 1])
top_cols[0].header("OpenAI Chat - Demo")
model = 'gpt-35-turbo'
choice = st.sidebar.selectbox(
    "Enterprise ChatGPT mode", ["ChatGPT", "Azure Cognitive Search", "AOAI embeddings"])
st.subheader(choice)
st.markdown(
    """
    <style>

        div[data-testid="column"]:nth-of-type(2)
        {
            text-align: end;
        } 
    </style>
    """, unsafe_allow_html=True
)

container = st.container()


def start_conversation():
    endpoint = "https://directline.botframework.com/v3/directline/conversations"
    response = requests.post(endpoint, headers=st.session_state['headers'])
    if response.status_code == 201:
        res = response.json()
        st.session_state['conversationId'] = res['conversationId']

        resp = GET()


def POST(payload):
    API_URL = f"https://directline.botframework.com/v3/directline/conversations/{st.session_state['conversationId']}/activities"

    response = requests.post(
        API_URL, headers=st.session_state['headers'], json=payload)
    return response.json()


def GET():
    API_URL = f"https://directline.botframework.com/v3/directline/conversations/{st.session_state['conversationId']}/activities"
    response = requests.get(API_URL, headers=st.session_state['headers'])
    return response.json()


def org_query(payload):
    # st.write(st.session_state['activities'])
    with container:
        if 'activities' not in st.session_state:
            message(payload['text'], is_user=True,
                    avatar_style="adventurer-neutral", key="usr_0")
        else:
            # st.write(f"skip these indices: {st.session_state['skip_idx']}")
            for idx, turn in enumerate(st.session_state['activities']):
                if idx in st.session_state['skip_idx']:
                    continue
                if 'text' not in turn:
                    if 'attachments' in turn:
                        attachment = turn['attachments'][0]
                        with container:
                            message(attachment['content']['title'], key=str(
                                len(st.session_state['activities'])+1), seed="Felix")
                            btns = process_option_response(attachment)
                            msg_btns(btns)

                    else:
                        # st.write(turn)
                        message("抱歉，我無法回答這個問題", key=str(
                            len(st.session_state['activities'])+1), seed="Felix")
                        st.session_state['read_text'] = "抱歉，我無法回答這個問題"
                    continue
                if turn['from']['id'] != "user1":
                    if 'tone' in turn:
                        col1, col2 = st.columns(2)

                        col2.write(f"Prompt: 使用{turn['tone']}口吻改寫")

                    message(turn['text'], key=str(idx + 1),
                            seed="Felix", add_openai=True if turn.get('aoai') else False)
                else:
                    message(turn['text'], is_user=True, avatar_style="adventurer-neutral",
                            key="usr_" + str(idx + 1))
            message(payload['text'], is_user=True, avatar_style="adventurer-neutral", key="usr_" +
                    str(len(st.session_state['activities'])+1))
            pass

    POST(payload)
    resp = GET()
    act_len = len(st.session_state['activities'])
    new_activities = resp['activities'][act_len+1:]

    if 'attachments' in new_activities[-1]:
        try:
            new_activities[-1]['text'] += "\n" + \
                new_activities[-1]['attachments'][0]['content']['text']
        except Exception as e:
            print(e)
            pass

    # if "text" in new_activities[-1]:

    if new_activities[-1].get('text') == "No QnAMaker answers found." or new_activities[-1].get('text') == "未找到答案":
        st.session_state['skip_idx'] += [act_len, act_len + 1]
        POST(payload)
        resp = GET()
        act_len = len(st.session_state['activities'])
        new_activities = resp['activities'][act_len+1:]
        if 'attachments' in new_activities[-1]:
            try:
                new_activities[-1]['text'] += new_activities[-1]['attachments'][0]['content']['text']
            except Exception as e:
                print(e)
                pass

    if st.session_state['aoai_enrich']:

        with container:
            col1, col2 = st.columns(2)

            col2.write(f"Prompt: 使用{st.session_state['tone']}口吻改寫")

        for idx, act in enumerate(new_activities):
            if act['from']['id'] != 'user1' and 'text' in act:
                new_activities[idx]['text'] = aoai_enrichment(
                    act['text'])
                new_activities[idx]['tone'] = st.session_state['tone']
                new_activities[idx]['aoai'] = True
    st.session_state['activities'] += new_activities

    return resp


def create_prompt(messages, sys_msg):
    prompt = system_message_template.format(sys_msg)
    message_template = "\n<|im_start|>{}\n{}\n<|im_end|>"
    for message in messages:
        prompt += message_template.format(message['sender'], message['text'])
    prompt += "\n<|im_start|>assistant\n"
    return prompt


def summarize(prompt):
    if model == "gpt-35-turbo":
        message = [{"sender": "user", "text": f"""
        '''
        {prompt}
        '''

        請用專業的口吻來簡化以上文字，且不要用本文開頭

        '''
        """}]
        summary = openai.Completion.create(
            engine="gpt-35-turbo",
            prompt=create_prompt(message, "你是專門負責摘要的機器人"),
            temperature=0.7,
            max_tokens=800,
            top_p=0.91,
            stop=["<|im_end|>"])["choices"][0]["text"]
       # st.success("done")
    else:

        augmented_prompt = f"請用以專業的口吻摘要這段文字: {prompt}"

        summary = openai.Completion.create(
            engine="davinci003",
            prompt=augmented_prompt,
            temperature=.5,
            max_tokens=2000,
        )["choices"][0]["text"].replace("\n", "")

    return summary


def aoai_enrichment(ans):
    augmented_prompt = f"請用中文以{st.session_state['tone']}的口吻改寫這段文字: {ans}"
    summary = openai.Completion.create(
        engine="davinci003",
        prompt=augmented_prompt,
        temperature=.5,
        max_tokens=2000,
    )["choices"][0]["text"].replace("\n", "")
    # st.write(summary)

    return summary


def aoai_query(query, mode, aoai_enrichment):
    if aoai_enrichment:
        docs = search_docs(query, mode, top_n=3)
        return summarize("".join(docs))
    else:
        docs = search_docs(query, mode, top_n=1)
        return docs[0]


def get_text():
    input_text = st.text_input("", "", key="text", label_visibility='hidden')
    return input_text


def clear_text():
    st.session_state['text'] = ""

# search through the reviews for a specific product


def search_docs(user_query, mode, top_n=3):
    if user_query != "":
        embedding = get_embedding(
            user_query,
            engine="text-similarity-curie-001"
        )
        st.write("Embedding Length: 4096")
        st.write("Sample of embeddings")
        st.write(embedding[:5])

        st.session_state['qa_df']["similarities"] = st.session_state['qa_df'][f'{mode[:-1]}_embedding'].apply(
            lambda x: cosine_similarity(x, embedding))

        res = (
            st.session_state['qa_df'].sort_values(
                "similarities", ascending=False)
            .head(top_n)
        )
        if top_n == 1:
            st.session_state['top_score'] = res.iloc[0]['similarities']
            st.session_state['top_q'] = res.iloc[0]['Question']
        return res['Answer'].to_list()


def load_data(mode):
    st.session_state['qa_df'] = pd.read_csv("./ASUS_KB.csv")
    tmp = np.load(f"./{mode[:-1]}_embeddings.npy")
    st.session_state['qa_df'][f'{mode[:-1]}_embedding'] = tmp.tolist()


def chatgpt_query(query):
    message = [{"sender": "user", "text": query}]
    response = openai.Completion.create(
        engine="gpt-35-turbo",
        prompt=create_prompt(
            message, "你是一個公共衛生專家，請回答這個領域的正確訊息，如果不確定就說'我無法回答'"),
        temperature=0.7,
        max_tokens=800,
        top_p=0.91,
        stop=["<|im_end|>"])["choices"][0]["text"]
    return response


def process_option_response(res):
    res = res['content']['buttons']
    return [btn['value'] for btn in res]


def ask(question):

    org_query({
        "type": "message",
        "from": {
            "id": "user1"
        },
        "text": question.target.value
    })
    with container:
        message(st.session_state['activities'][-1]['text'],
                key=str(len(st.session_state['activities'])+1), seed="Felix", add_openai=True if st.session_state['activities'][-1].get("aoai") else False)


def msg_btns(btns):
    with elements(str(uuid.uuid4())):
        mui.Stack(
            html.input(
                type="button",
                value=btns[0],
                onClick=ask,
                style={
                    "backgroundColor": "rgba(146, 140, 139, 0.5)",
                    "color": "white",
                    "borderRadius": "10px",
                    "padding": "10px 15px",
                    "border": "none"
                }),
            html.input(
                type="button",
                value=btns[1],
                onClick=ask,
                style={
                    "backgroundColor": "rgba(146, 140, 139, 0.5)",
                    "color": "white",
                    "borderRadius": "10px",
                    "padding": "10px 15px",
                    "border": "none"
                }),
            html.input(
                type="button",
                value=btns[2],
                onClick=ask,
                style={
                    "backgroundColor": "rgba(146, 140, 139, 0.5)",
                    "color": "white",
                    "borderRadius": "10px",
                    "padding": "10px 15px",
                    "border": "none"
                }),
            width="fit-content",
            spacing=1,
            sx={
                "marginLeft": "5%",
            }
        )


def read():
    clear_text()
    speech_synthesis_result = st.session_state['speech_synthesizer'].speak_text_async(
        st.session_state['read_text']).get()
    if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized for text [{}]".format(
            st.session_state['read_text']))
    elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        print("Speech synthesis canceled: {}".format(
            cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(
                    cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")


def main():
    globals()
    name, authentication_status, username = authenticator.login(
        'Login', 'main')
    if authentication_status:
        #     if 'cs_key' in st.session_state:
        #         speech_config = speechsdk.SpeechConfig(
        #             subscription=st.session_state['cs_key'], region="eastus")
        #         audio_config = speechsdk.audio.AudioOutputConfig(
        #             use_default_speaker=True)
        #         speech_config.speech_synthesis_voice_name = 'zh-CN-YunxiNeural'

        #         st.session_state['speech_synthesizer'] = speechsdk.SpeechSynthesizer(
        #             speech_config=speech_config)
        with top_cols[1]:
            st.write("   ")
            authenticator.logout('Logout', 'main')

        st.session_state['headers'] = {
            "Authorization": f"Bearer {st.secrets['api']['qna_bot_key']}"}

        if 'skip_idx' not in st.session_state:
            st.session_state['skip_idx'] = []

        if choice == "Azure Cognitive Search":
            st.session_state['aoai_enrich'] = st.sidebar.checkbox(
                "AOAI Tone-Enhanced")

            if st.session_state['aoai_enrich']:
                st.session_state['tone'] = st.sidebar.selectbox(
                    "選擇一種口吻", ['專家', '小孩', '活潑', '工程師', '業務'], key="")

            if 'activities' not in st.session_state:
                st.session_state['activities'] = []

            with container:
                message(
                    "我是ASUS FAQ機器人，有什麼我可以協助您的嗎?", key="0", seed="Felix")

            with st.form("user query"):
                st.session_state["user_input"] = st.text_input("有什麼問題呢?")
                submitted = st.form_submit_button("送出")

            if submitted:
                if 'conversationId' not in st.session_state:
                    start_conversation()

                org_query({
                    "type": "message",
                    "from": {
                        "id": "user1"
                    },
                    "text": st.session_state["user_input"]
                })

                if 'activities' in st.session_state:
                    with container:
                        try:
                            message(st.session_state['activities'][-1]['text'],
                                    key=str(len(st.session_state['activities'])+1), seed="Felix", add_openai=True if st.session_state['activities'][-1].get("aoai") else False)
                            st.session_state['read_text'] = st.session_state['activities'][-1]['text']

                        except Exception as e:
                            if 'attachments' in st.session_state['activities'][-1]:
                                attachment = st.session_state['activities'][-1]['attachments'][0]
                                with container:
                                    message(attachment['content']['title'], key=str(
                                        len(st.session_state['activities'])+1), seed="Felix")
                                    btns = process_option_response(attachment)
                                    msg_btns(btns)

                            else:
                                message("抱歉，我無法回答這個問題", key=str(
                                    len(st.session_state['activities'])+1), seed="Felix")
                                st.session_state['read_text'] = "抱歉，我無法回答這個問題"

            elif 'activities' in st.session_state:
                with container:
                    for idx, turn in enumerate(st.session_state['activities']):
                        if idx in st.session_state['skip_idx']:
                            continue
                        if 'text' not in turn:
                            if 'attachments' in turn:
                                attachment = turn['attachments'][0]
                                with container:
                                    message(attachment['content']['title'], key=str(
                                        len(st.session_state['activities'])+1), seed="Felix")
                                    btns = process_option_response(attachment)
                                    msg_btns(btns)

                            else:
                                message("抱歉，我無法回答這個問題", key=str(
                                    len(st.session_state['activities'])+1), seed="Felix")
                                st.session_state['read_text'] = "抱歉，我無法回答這個問題"
                            continue
                        if turn['from']['id'] != "user1":
                            if turn.get('aoai'):
                                col1, col2 = st.columns(2)

                                col2.write(f"Prompt: 使用{turn['tone']}口吻改寫")
                            # st.write(turn)
                            message(turn['text'], key=str(
                                idx + 1), seed="Felix", add_openai=True if turn.get('aoai') else False)
                            if idx == len(st.session_state['activities']) - 1:
                                st.session_state['read_text'] = st.session_state['activities'][-1]['text']
    #                             st.button("speech", on_click=read)
                        else:
                            message(turn['text'], is_user=True, avatar_style="adventurer-neutral",
                                    key="usr_" + str(idx + 1))
        elif choice == "AOAI embeddings":
            qa_mode = "questions"
            st.sidebar.write("""
            藉由Text Embeddings來強化搜尋比對的能力
            """)
            st.session_state['aoai_enrich'] = st.sidebar.checkbox(
                "AOAI Enrichment")
            if "qa_df" not in st.session_state:
                load_data(qa_mode)
                st.session_state['qa_mode'] = qa_mode

            if qa_mode != st.session_state.get('qa_mode'):
                load_data(qa_mode)
                st.session_state['qa_mode'] = qa_mode

            if 'generated' not in st.session_state:
                st.session_state['generated'] = []

            if 'past' not in st.session_state:
                st.session_state['past'] = []

            with container:
                message(
                    "我是整合了AOAI的ASUS FAQ機器人，有什麼我可以協助您的嗎?", key="0", seed="Aneka", is_openai=True)

                if st.session_state['generated']:

                    for i in range(len(st.session_state['generated'])):
                        message(st.session_state['past'][i],
                                is_user=True, avatar_style="adventurer-neutral", key=str(i+1) + '_user')
                        message(st.session_state["generated"]
                                [i], key=str(i+1),  seed="Aneka", is_openai=True)

            with st.form("user query"):
                st.session_state["user_input"] = st.text_input("有什麼問題呢?")
                submitted = st.form_submit_button("送出")

            if submitted:
                with container:
                    st.session_state.past.append(
                        st.session_state["user_input"])
                    message(st.session_state["user_input"], is_user=True, avatar_style="adventurer-neutral", key=str(
                        len(st.session_state.past))+"_user")

                res = aoai_query(st.session_state["user_input"], qa_mode,
                                 st.session_state['aoai_enrich'])

                with container:
                    st.session_state.generated.append(res)
                    message(res, key=str(
                        len(st.session_state.generated)), seed="Aneka", is_openai=True)
                    if not st.session_state['aoai_enrich']:
                        st.write(
                            f"Selected Question: {st.session_state['top_q']}")
                        st.write(
                            f"Similarity Score: {st.session_state['top_score']:.2f}")
        else:
            if 'chatgpt_generated' not in st.session_state:
                st.session_state['chatgpt_generated'] = []

            if 'chatgpt_past' not in st.session_state:
                st.session_state['chatgpt_past'] = []

            with container:
                message(
                    "我是AOAI提供的ChatGPT機器人，有什麼我可以協助您的嗎?", key="0", seed="Aneka", is_openai=True)

                if st.session_state['chatgpt_generated']:

                    for i in range(len(st.session_state['chatgpt_generated'])):
                        message(st.session_state['chatgpt_past'][i],
                                is_user=True, avatar_style="adventurer-neutral", key=str(i+1) + '_user')
                        message(st.session_state["chatgpt_generated"]
                                [i], key=str(i+1),  seed="Aneka", is_openai=True)

            with st.form("user query"):
                st.session_state["user_input"] = st.text_input("有什麼問題呢?")
                submitted = st.form_submit_button("送出")
            if submitted:
                with container:
                    st.session_state.chatgpt_past.append(
                        st.session_state["user_input"])
                    message(st.session_state["user_input"], is_user=True, avatar_style="adventurer-neutral", key=str(
                        len(st.session_state.chatgpt_past))+"_user")

                res = chatgpt_query(
                    st.session_state["user_input"])

                with container:
                    st.session_state.chatgpt_generated.append(res)
                    message(res, key=str(
                        len(st.session_state.chatgpt_generated)), seed="Aneka", is_openai=True)

    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')


if __name__ == "__main__":
    main()


# if st.session_state['generated']:

#     for i in range(len(st.session_state['generated'])-1, -1, -1):
#         message(st.session_state["generated"][i], key=str(i))
#         message(st.session_state['past'][i],
#                 is_user=True, key=str(i) + '_user')
