import weaviate
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime, timedelta

# langchain_community
from langchain_community.vectorstores import Weaviate
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ChatMessageHistory

# from app.core.config import OPENAI_KEY, OPENAI_ORGANIZATION, WV_KEY
from app.core.singleton import Singleton

# openai.api_key = OPENAI_KEY
# openai.organization = OPENAI_ORGANIZATION
# weaviate_client = weaviate.Client(url = "http://ai.diplomacy.edu:8080", auth_client_secret=weaviate.AuthApiKey(WV_KEY))

"""
Repacks the chat history by converting the memory and system message into a ChatMessageHistory object.

Parameters:
memory (list): A list of chat messages.
system_message (str): The system message.

Returns:
ChatMessageHistory: The repacked chat history.

"""
# context: called by chat_service.py to repack the chat history
# description: this is filled by state (user_type + system prompt) messages
def repack_chat_history(memory, system_message):
    sys_message = SystemMessage(system_message)
    history = ChatMessageHistory()
    history.add_message(sys_message)
    for message in memory:
        if message['role'] == 'user':
            history.add_user_message(message['content'])
        else:
            history.add_ai_message(message['content'])
    return history


"""
Repacks the chat history by converting the memory and system message into a ChatMessageHistory object.

Parameters:
memory (list): A list of chat messages.
system_message (str): The system message.

Returns:
ChatMessageHistory: The repacked chat history.
"""
def cleanup_conversation_history():
    
    conversation_history = Singleton().conversation_history
    
    cutoff_time = datetime.now() - timedelta(minutes=30)
    for user_id in list(conversation_history.keys()):
        for chatbot_id in list(conversation_history[user_id].keys()):
            conversation = conversation_history[user_id]
            if len(conversation) > 0 and conversation[0]['timestamp'] < cutoff_time:
                del conversation_history[user_id]


'''
Mockups
'''

"""
Returns a mockup response for the chat route.

Returns:
    dict: A dictionary containing the mockup response with the following keys:
        - "answer" (str): The answer to the chat query.
        - "sources" (list): A list of dictionaries representing the sources of the answer.
            Each dictionary contains the following keys:
            - "text" (str): The text of the source.
            - "date" (str): The date of the source.
            - "title" (str): The title of the source.
            - "url" (str): The URL of the source.
            - "authors" (list): A list of authors of the source.
"""
def get_chat_route_response_mockup():
    return {
        "answer": "Diplomacy education refers to the academic study and training related to the practice of diplomacy. It involves learning about the principles, practices, and methods of conducting international relations, negotiations, and communication between countries. Diplomacy education can cover a wide range of topics, including the history of diplomacy, international law, conflict resolution, cultural understanding, and the role of diplomats in promoting peace and cooperation among nations. It can be pursued through formal academic programs, courses, workshops, and training sessions designed to prepare individuals for careers in diplomacy and international affairs.",
        "sources": [
            {
            "text": "What is Diplomacy? Towards Education Diplomacy?\nhttp://www.diplomacy.edu/blog/what-diplomacy-towards-education-diplomacy\n\n'New diplomacy' has become somewhat of a buzzword. In its current form it mainly describes new actors becoming more visible in the diplomatic process. We have also seen new terms such as health diplomacy being used more frequently. Here, I am wondering about the potential of so-called education diplomacy. \nDiploFoundation and the Association for Childhood Education International (ACEI) are working together to develop an online course which is scheduled to run this fall. Having been inivited to speak at ACEI's Global Institute for Education Diplomacy (March 5-8, Washington, DC), I presented some thoughts on the nature of diplomacy and the emerging concept of education diplomacy. In the following, you can find my remarks at the panel. Further comments are more than welcome. \n\nGlobal Institute for Education Diplomacy. General session panel: What is diplomacy? Left to right: Diane Whitehead (ACEI), Karen Dickman (Institute for Multitrack Diplomacy), Katharina Hone (DiploFoundation). Photo posted by @ACEI_info\nSomeone once said that “[e]ducation is the most powerful weapon which you can use to change the world.” I am sure most of you know which important global leader I’m quoting here. Let us take this as a first point of motivation when critically engaging Education Diplomacy.\nThe first and most obvious question to raise is: what is diplomacy? For a scholar of diplomacy, one well-learned and often rehearsed answer immediately springs to mind: diplomacy is the management of international relations by negotiation. It is undertaken by designated state officials who enjoy privileges and immunities when abroad.\nWhile text book definitions such as this one are designed to make the world look simple, we all know that it is rarely that simple. But moreover, for practitioners and those interested in change, it is paramount to not only question the received wisdom, but to eventually move beyond it. The parameters of diplomacy are enshrined in international law, most importantly in the 1961 Vienna Convention on Diplomatic Relations. However, first and foremost, diplomacy is a practice. And it is only through its practice that diplomacy comes about, is sustained – but also changed. This is an occasion to reflect on this practice and its changing nature.\nOver the last two decades we could see many examples that can be taken as a challenge to the definition of diplomacy I just gave. We have seen the emergence of non-state actors on the diplomatic playing field and the rise of so-called soft issues – such as health, the environment, and education to name a few. Some have coined this ‘new diplomacy.’\nOne of the questions in this regard is to what extent non-state actors can influence global agendas, decisions, and implementation. This needs to be carefully analysed on a case-by-case basis. And I would like to raise a first point of caution here. While non-officials and non-state actors have become more prominent, more visible in the diplomatic process, we need to wonder to what extent they influence decisions. Non-state actors are often associated with technical and specialised expertise. They are seen as partners in setting the agenda and in implementing decisions. However, we need to wonder: to what extent can decisions be influenced by these ‘new diplomats.’ ‘New diplomacy’ can all too easily become a buzz word that hides that not much has changed, that the game in town is essentially still the same – a game dominated by officials representing powerful states. That is why we need to be careful not to mistake facades that are painted in friendlier colours with real change.\nKeeping that in mind, there is a second highly important point of departure for those interested in education diplomacy. In addition to asking ‘what is diplomacy,’ we need to start by asking about our motivations and goals with regard to developing and using education diplomacy.\nI would argue that education diplomacy has a very strong normative dimension that takes us quickly beyond national interests, narrowly conceived. We cannot escape this normative dimension when speaking about education diplomacy and frankly we should not try. To do so would hollow out the concept and practice before we have even begun to embrace it.\nI am sure many of you will agree that education is the foundation, the foundation for many other positive achievements. Education is a fundamental human right.",
            "date": "2015-03-11 20:14:33",
            "title": "What is Diplomacy? Towards Education Diplomacy?",
            "url": "http://www.diplomacy.edu/blog/what-diplomacy-towards-education-diplomacy",
            "authors": []
            },
            {
            "text": "Types of diplomacy\nhttp://www.diplomacy.edu/topics/types-of-diplomacy\n\nMethods and tools for conducting diplomacy: bilateral and multilateral diplomacy,  public diplomacy , and metaverse diplomacy. \n\n\n \n[supsystic-tables id=34]\n \r\n\r\n[caption id=\"attachment_102040\" align=\"aligncenter\" width=\"773\"] Books on various types of diplomacy – Library of Jovan Kurbalija[/caption]Education diplomacy There are 136 uses of the term diplomacy in three contexts:\n\nGeopolitics and geo-economics (dollar diplomacy, gunboat diplomacy)\n\nTopics areas addressed by diplomacy: digital diplomacy, development diplomacy, economic diplomacy, cyber diplomacy, AI diplomacy, energy diplomacy, health diplomacy,  science diplomacy, sport diplomacy, climate diplomacy, education diplomacy, etc.\n\n\nMethods and tools for conducting diplomacy: bilateral and multilateral diplomacy,  public diplomacy , and metaverse diplomacy. \n\n\n \n[supsystic-tables id=34]\n \r\n\r\n[caption id=\"attachment_102040\" align=\"aligncenter\" width=\"773\"] Books on various types of diplomacy – Library of Jovan Kurbalija[/caption]Education diplomacy is the use of education as a tool to promote international relations and foster mutual understanding between countries. It involves the exchange of students, faculty, and ideas between countries, as well as the development of educational initiatives that promote global understanding and collaboration. Education diplomacy can also involve the use of educational resources to support international development goals.",
            "date": "",
            "title": "Types of diplomacy",
            "url": "http://www.diplomacy.edu/topics/types-of-diplomacy",
            "authors": []
            },
            {
            "text": "What is Diplomacy? Towards Education Diplomacy?\nhttp://www.diplomacy.edu/blog/what-diplomacy-towards-education-diplomacy\n\nThat is why we need to be careful not to mistake facades that are painted in friendlier colours with real change.\nKeeping that in mind, there is a second highly important point of departure for those interested in education diplomacy. In addition to asking ‘what is diplomacy,’ we need to start by asking about our motivations and goals with regard to developing and using education diplomacy.\nI would argue that education diplomacy has a very strong normative dimension that takes us quickly beyond national interests, narrowly conceived. We cannot escape this normative dimension when speaking about education diplomacy and frankly we should not try. To do so would hollow out the concept and practice before we have even begun to embrace it.\nI am sure many of you will agree that education is the foundation, the foundation for many other positive achievements. Education is a fundamental human right. At the same time education is fundamental to human rights – their enjoyment, their defence, and calls for their actualization. Education is part of many development cooperation initiatives, yet it is also the foundation for development. I am sure this hardly needs to be emphasized before an audience like you. Rather, my point is that this normative dimension, and maybe the more institution-specific and even personal goals we have, cannot be written out of an understanding of education diplomacy. On the contrary, they need to be embraced. But let me also point out that what I have just said does not square well with the more traditional perspective on diplomacy I alluded to earlier.\nSo far, we have two ingredients: an understanding of diplomacy and an understanding of the normative dimension of education. Now, I would like to add a suggestion why bringing the two together matters. Hence, why education diplomacy matters.\nFrom my perspective, it matters because decisions are taken at a global level that influence the work on the ground, that influence the possibilities within classrooms everyday. The Millennium Development Goals are one example familiar to many. The second Millennium Development Goal sets the ideal of universal primary education. This specific focus of MDG 2 on primary education has profound consequences. Many efforts, many successful efforts, have been made to increase enrolment rates and access to primary education – especially in the global South. However, critical voices argue that this came at a cost. The cost was that the focus shifted away from the quality of education to the quantitative measure of enrolment rates. Further, education after the primary level tended to be put on the back burner. This means that as we move from the Millennium Development Goals to the Sustainable Development Goals in the post-2015 development agenda, these problems need to be addressed. To me, it seems that education diplomacy will be crucial here.\nThis brings me to my last point. I would like to conclude by asking the most important question: what is education diplomacy?\nOnce we start looking, as I began to do more closely in October last year, we begin to see education and its relevance everywhere. If we are interested in developing the concept and practice of education diplomacy, this is a challenge.  Everywhere very quickly can mean nowhere. This is where I would like to add my second point of caution for today. When we speak of education diplomacy we need to use the term carefully and deliberately. It is clear that the term goes beyond traditional understandings of diplomacy. Yet, we need to take care in delineating it in some way.\nOne way to start is by listing practices we can consider education diplomacy.\nSuggestions I came up with include:\n•   activities of the United Nations Educational, Scientific and Cultural Organization (UNESCO) and the United Nations Children’s Fund (UNICEF),\n•   various world summits such as the World Summit for Children [and the World Conference on Education For All in 1990, the 1993 World Conference on Human Rights, and the 2000 World Education Forum],\n•   the negotiation and implementation of the Millennium Development Goals (especially goal two calling for universal primary education) and the work towards the Sustainable Development Goals (especially goal four),\n•   negotiations within the World Trade Organization (WTO), in particular as they relate to education as part of the General Agreement on Trade in Services (GATS),\nI am sure, from your own experience, there is a plethora of examples to be added here. It will be great to debate these and enrich the list over the next days.",
            "date": "2015-03-11 20:14:33",
            "title": "What is Diplomacy? Towards Education Diplomacy?",
            "url": "http://www.diplomacy.edu/blog/what-diplomacy-towards-education-diplomacy",
            "authors": []
            },
            {
            "text": "Diplomatic Education\nhttp://www.diplomacy.edu/resource/diplomatic-education\n\nWhat is the professional expertise needed by a diplomat? One should not be surprised that understanding of societal affairs and economics is more important as a knowledge base than the theory of international relations.\nKishan S. Rana, DIPLOMAT AND AUTHOR, INDIA  \nIt begins:\nIn most countries, those selected for the diplomatic service are elites. This does not refer to their social background — in fact in almost all countries a democratization process is evident, in terms of the economic groups and the educational institutions to which the new entrants belong. They are elites because behind each young man or woman who wins the coveted appointment, stand dozens, or in some countries, even several hundreds, of unsuccessful applicants. Despite all the diversification in job opportunities that has taken place in our globalizing world, and the opening up of career avenues that did not exist a decade or two back, representing one’s country abroad remains a coveted honor, attracting the best and the brightest in virtually every country that has an open, competitive selection process.\nWhat kind of higher education is the best preparation for a career in a diplomatic service? Is there a particular kind of discipline that is best suited to produce envoys? What are the needs for professional training for diplomats, at the stage of induction, and later on, as the individual’s career progresses?",
            "date": "2021-06-17 08:39:42",
            "title": "Diplomatic Education",
            "url": "http://www.diplomacy.edu/resource/diplomatic-education",
            "authors": []
            }
        ]
    }
    