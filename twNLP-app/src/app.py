import streamlit as st
from views.components.spinner import dowload_ckip_package, download_cwn_drivers
import pandas as pd
import requests 
import bs4
from snownlp import SnowNLP


def run_app(ckip_nlp_models, cwn_upgrade) -> None:
    # need to download first because CWN packages will first check whether
    # there is .cwn_graph folder in the root directory.
    download_cwn_drivers(cwn_upgrade)
    dowload_ckip_package(ckip_nlp_models)

    from views.components.sidebar import visualize_side_bar
    from views.containers import display_cwn, display_ckip, display_data_form

    st.title("NLP app for PTT")
    st.write("這是一個針對PTT語料的 情緒分析｜中文NLP管線處理🔎")
    st.image("/Users/joannechi/nlpWeb/myApp/nlpweb/nlp_assignment_1/img/Mo-PTT-Logo.png", width=200)

    #menu = ["Text","Sentences"]
    #choice = st.sidebar.selectbox("Menu",menu) 


    #spectra = st.file_uploader("upload your file", type={"csv", "txt"})
    #if spectra is not None:
    #    spectra_df = pd.read_csv(spectra) #讀取csv
    #    st.write(spectra_df)

    #~~web crawler~~
    st.subheader("PTT Crawler 🐛")
    st.text('目前看板有：HatePolitics｜Gossiping｜Military｜Stock')
    selected = st.selectbox('請選擇看板：', 
      ['HatePolitics', 'Gossiping','Military','Stock']) 
    if selected=='HatePolitics':
        URL = "https://www.ptt.cc/bbs/HatePolitics/index.html"
    elif selected=='Gossiping':
        URL = "https://www.ptt.cc/bbs/Gossiping/index.html"
    elif selected=='Military':
        URL = "https://www.ptt.cc/bbs/Military/index.html"
    else:
        URL = "https://www.ptt.cc/bbs/Stock/index.html"

    my_headers = {'cookie': 'over18=1;'}
    response = requests.get(URL, headers = my_headers)
    soup = bs4.BeautifulSoup(response.text,"html.parser")
    list_results=[]
    for t in soup.find_all('div','title'):
        find_a=t.find('a')
        find_href="https://www.ptt.cc"+find_a.get("href")
        title=t.text
        results={
            "title":title,
            "url":find_href
        }
        list_results.append(results)        
    my_df=pd.DataFrame(list_results)
    print(my_df)
    st.write(my_df)
    #~~web crawler~~

    #~~sentiment analysis~~
    st.subheader("情緒分析")
    with st.form(key="nlpForm"):
        raw_text=st.text_area("請輸入句子✏️")
        submit_button=st.form_submit_button(label="確定")
        
    if submit_button:
        
        st.info("sentiment")
        sentiment=SnowNLP(SnowNLP(raw_text).han) #轉簡體
        sentiment_han=sentiment.sentiments
        st.write(sentiment_han)

            #emoji
        if sentiment_han>0:
            st.markdown("Sentiment:: Positive :smiley: ")
        elif sentiment_han<0:
            st.markdown("Sentiment:: Negative :angry: ")
        else:
            st.markdown("Sentiment:: Neutral :neutral: ")



        #with col2:
            #st.info("category")
            #category=SnowNLP(SnowNLP(raw_text).han) #轉簡體
            #category_han=list(category.tags)
            #st.write(category_han)

    #~~sentiment analysis~~

    st.subheader("中文 NLP 管線處理")

    input_data = display_data_form()
    model, pipeline, active_visualizers = visualize_side_bar(ckip_nlp_models) 
                                        #return model_options, pipeline_options, active_visualizers

    display_factories = {"CKIP": display_ckip, "CWN": display_cwn}

    if "input_data" in st.session_state:
        display_factories[pipeline](
            model, active_visualizers, st.session_state["input_data"]
        )


if __name__ == "__main__":
    ckip_nlp_models = ["bert-base", "albert-tiny", "bert-tiny", "albert-base"]
    run_app(ckip_nlp_models, cwn_upgrade=False)
