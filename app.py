import streamlit as st
from markitdown import MarkItDown

# from model import generate_easy_read, load_model
from model_aws import generate_easy_read, count_tokens
from model_swa import translate
from evaluate import evaluate


MODEL_OPTIONS = {
    "English": "meta.llama3-3-70b-instruct-v1:0",
    "Swahili": "swahili"
}

MAX_ALLOWED_TOKENS = 4096

default_prompt_path = "./prompts/prompt_2.txt"

with open(default_prompt_path, "r") as f:
    default_system_prompt = f.read()  # 392 tokens

md = MarkItDown(enable_plugins=False)

def main():

    st.set_page_config(
        layout="wide",
        page_title="EasyRead",
        page_icon="📝",
        initial_sidebar_state="collapsed",
    )

    hide_streamlit_style = """
    <style>
        header {visibility: hidden;}
        .streamlit-footer {display: none;}
        .st-emotion-cache-uf99v8 {display: none;}
    </style>
    """

    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.markdown(
        """
        <div style="display: flex; justify-content: center;">
            <img src="https://i.ibb.co/1fzM5Gbn/fenix-logo.jpg" width="100">
        </div>
    """,
        unsafe_allow_html=True,
    )

    st.title("Convert Content to EasyRead")

    col1, col2 = st.columns([3, 3])

    with col1:
        user_input = st.text_area("Input:", height=250)
        uploaded_file = st.file_uploader("Or upload a file", type=["pdf", "doc", "docx", "xlsx"])
        if uploaded_file is not None:
            result = md.convert(uploaded_file)
            user_input = result.text_content

        selected_model = st.selectbox(
            "Select Model:", list(MODEL_OPTIONS.keys()), index=0
        )

        with st.expander("More"):
            system_prompt = st.text_area("System Prompt", value=default_system_prompt, height=400)
            user_context = st.text_area("Background Information (optional):", height=100)

            st.divider()
            st.caption("Convert complex text into a simpler, easier-to-read format with clear guidelines.")
            st.caption("This tool enhances accessibility by structuring information in a clear and straightforward way.")
            st.markdown("The platform follows the [Easy Read](https://www.inclusion-europe.eu/wp-content/uploads/2017/06/EN_Information_for_all.pdf) guidelines for better readability.")

        model_id = MODEL_OPTIONS[selected_model]

        st.write("")  # Spacer
        if st.button("Convert", use_container_width=True):
            user_input_token = count_tokens(user_input)
            if user_input.strip() and user_input_token > MAX_ALLOWED_TOKENS:
                st.error(
                    f"Input text is too large: {user_input_token}. Max tokens allowed is {MAX_ALLOWED_TOKENS}. Please reduce the input size."
                )
                return

            if user_input.strip():
                with st.spinner("Converting..."):
                    try:
                        output = None
                        if model_id == "swahili":
                            # output_orig = generate_easy_read(
                            #     user_input, model_id="meta.llama3-3-70b-instruct-v1:0", context_info=user_context, system_prompt=system_prompt
                            # )
                            # output = translate(output_orig, source_language="eng", target_language="swa")
                            prompt = system_prompt + user_context if user_context is not None else system_prompt
                            custom_prompt = "Please follow the above instruction. Regardless of whether the content is in English or Swahili, your response should only be in Swahili only. Only return the easyread in Swahili. Content:"
                            output = get_easyread(
                                text=user_input,
                                prompt=f"{prompt}\n{custom_prompt}"
                            )
                        else:
                            output = generate_easy_read(
                                user_input, model_id=model_id, context_info=user_context, system_prompt=system_prompt
                            )
                        
                        with col2:
                            st.write("\n\n")
                            # st.write(output)
                            st.markdown(f"<div style='font-size:16px;'>{output}</div>", unsafe_allow_html=True)
                            eval_result = evaluate(output)
                            st.json(eval_result)

                    except MemoryError:
                        st.error("Try reducing input size.")
            else:
                st.warning("Please provide a text to convert.")

    with col2:
        st.write("")
        st.write("")


if __name__ == "__main__":
    main()
