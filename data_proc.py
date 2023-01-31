import nltk
nltk.download('punkt')
nltk.download('stopwords')

import re
import requests
import csv
import pymorphy2
import numpy as np
import networkx as nx
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.cluster.util import cosine_distance


BASE_URL = r'https://api.hh.ru/vacancies?'
CLEANER = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')


def text_clean(raw_text):
    """To clear text from HTML tags"""

    clean_text = re.sub(CLEANER, '', raw_text)
    clean_text = clean_text.lower()
    clean_text = clean_text.replace(';', '.')

    return clean_text


def tokenize_words(raw_text):
    """To tokenize text by first removing
     punctuation marks and extra spaces"""

    marks_removed = re.sub(r'[^\w\s]', '',  raw_text)
    cleaned_text = re.sub(" +", " ", marks_removed)

    return nltk.word_tokenize(cleaned_text)


def tokenize_text(raw_text):
    """To tokenize all text by sentences
     and each sentence by words"""

    cleaned_text = text_clean(raw_text)
    sentence_tokens = sent_tokenize(cleaned_text)

    for i in range(len(sentence_tokens)):
        sentence = sentence_tokens[i]
        # clean_text = text_clean(sentence)
        token_words = tokenize_words(sentence)
        sentence_tokens[i] = token_words

    return sentence_tokens


def remove_stop_words(raw_list):
    """To remove stop words from the list
    (prepositions, particles, etc.)"""

    stop_words = stopwords.words('russian')
    filtered_list = []

    for word in raw_list:
        if word not in stop_words:
            filtered_list.append(word)

    return filtered_list


def morph_analyse(raw_list):
    """For morphological analysis of words
     and converting a list to a string"""

    morph = pymorphy2.MorphAnalyzer()
    morph_list = []

    for word in raw_list:
        parsed_word = morph.parse(word)[0].normal_form
        morph_list.append(parsed_word)

    morph_str = ' '.join(morph_list)

    return morph_str


def text_preprocessing(raw_text):
    """General text preprocessing function"""

    text_cln = text_clean(raw_text)
    text_tkn = tokenize_words(text_cln)
    text_stp = remove_stop_words(text_tkn)
    text_morph = morph_analyse(text_stp)

    return text_morph


def sentence_similarity(sent1, sent2, stop_words):
    """To vectorize two sentences"""

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for word in sent1:
        if word in stop_words:
            continue

        vector1[all_words.index(word)] += 1

    for word in sent2:
        if word in stop_words:
            continue

        vector2[all_words.index(word)] += 1

    return 1 - cosine_distance(vector1, vector2)


def build_similarity_matrix(sentences, stop_words):
    """To generate similar matrix across sentences"""

    sim_matrix = np.zeros((
        len(sentences),
        len(sentences)
    ))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue

            sim_matrix[idx1][idx2] = sentence_similarity(
                sentences[idx1],
                sentences[idx2],
                stop_words
            )

    return sim_matrix


def generate_summary(text, top_n=5):
    """To get the summarized text,
    indicate the number of required sentences"""

    stop_words = stopwords.words('russian')
    summarize_text = []

    sentences = tokenize_text(text)
    similarity_martix = build_similarity_matrix(sentences, stop_words)
    similarity_graph = nx.from_numpy_array(similarity_martix)
    scores = nx.pagerank(similarity_graph)

    ranked_sentence = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)),
        reverse=True
    )

    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))

    return ". ".join(summarize_text)


def get_vacancies(key='', only_with_salary=False, limit=20):
    """To get data in JSON with parameters
    ("key", "limit", "only_with_salary")
    by which the data will be filtered"""

    url = BASE_URL + f'&text={key}' \
                     f'&per_page={limit}' \
                     f'&only_with_salary={only_with_salary}'

    response = requests.get(url)
    vacancies = response.json()

    return vacancies


def get_vacancy(id):
    """To get vacancy by id"""

    url = f'https://api.hh.ru/vacancies/{id}'

    response = requests.get(url)
    vacancy = response.json()

    return [
        vacancy['name'],
        vacancy['area']['name'],
        vacancy['employer']['name'],
        vacancy['salary']['from'],
        vacancy['salary']['to'],
        vacancy['salary']['currency'],
        vacancy['description'],
        vacancy['specializations'][0]['name'],
        vacancy['specializations'][0]['profarea_name'],
        vacancy['schedule']['name'],
        vacancy['experience']['name'],
    ]


def write_data_csv(key=''):
    """To write all received data to a csv file.
    Data can be filtered by "key" parameter"""

    now = datetime.now()
    date = now.strftime("%Y-%m-%d-%H%M%S")

    file_name = f'vacancies{date}.csv'

    with open(file_name, 'w', encoding='UTF8', newline='') as f:

        writer = csv.writer(f)
        writer.writerow([
            'name',
            'area',
            'employer',
            'salary_from',
            'salary_to',
            'currency',
            'description',
            'specialization',
            'profarea',
            'schedule',
            'experience',
        ])

        for page in range(20):
            url = BASE_URL + f'&text={key}' \
                             f'&page={page}' \
                             f'&per_page=100' \
                             f'&only_with_salary=true'

            response = requests.get(url)
            vacancies = response.json()

            for item in vacancies['items']:
                vacancy = get_vacancy(item['id'])

                writer.writerow(vacancy)

    return file_name


if __name__ == '__main__':
    text_raw = '<p><strong>Требования:</strong></p> <ul> <li>Опыт работы на аналогичной позиции – от 1 года;' \
               '</li> <li>Опыт ведения коммерческой базы предприятия;</li> <li>Активная жизненная позиция‚ ' \
               'грамотная речь‚ коммуникабельность;</li> <li>Высокая самоорганизация, лидерские качества, ' \
               'аналитический склад ума;</li> <li>Уверенное пользование 1С</li> </ul> <p><strong>Обязанности:' \
               '</strong></p> <ul> <li>Самостоятельное ведение бухгалтерского, налогового учета в полном' \
               ' объеме в соответствии с требованиями законодательства КР;</li> <li>Формирование и сдача ' \
               'бухгалтерской, налоговой, статистической отчетности;</li> <li>Проверка и подписание всей ' \
               'первичной документации (кассовые документы. авансовые отчеты, инвентаризации, доверенности' \
               ' и тд);</li> <li>Формирование прихода ТМЦ в базу 1С, перемещение по складам;</li> <li>' \
               'Ведение учета движения ТМЦ по документам поступления, отгрузки, перемещения в программе 1С;' \
               '</li> <li>Ценообразование товаров;</li> <li>Учет и контроль дебиторской и кредиторской' \
               ' задолженности;</li> <li>Ведение кассовых и банковских операций, проверка авансовых' \
               ' отчетов, ведение учета материальных ценностей;</li> <li>Контроль за основными и иными' \
               ' материальными ценностями организации;</li> <li>Контроль и участие в процессе инвентаризации' \
               ' материального имущества; выявление итогов инвентаризации и анализ недостачи;</li> <li>' \
               'Анализ финансово-хозяйственной деятельности предприятия;</li> <li>Учет труда и начисление' \
               ' заработной платы сотрудникам, своевременное отражение всех операций в бухгалтерском учете;' \
               '</li> <li>Формирование финансовой отчетности;</li> <li>Выполнение актов -сверок с ' \
               'контрагентами</li> <li>Интернет-банкинг и взаимодействие с банками;</li> <li>Ведение' \
               ' кадровой документации;</li> <li>Выполнение отдельных поручений руководителя;</li> ' \
               '</ul> <p><strong>Условия:</strong></p> <ul> <li>Официальное трудоустройство по ТК;</li>' \
               ' <li>Местонахождение офиса - центр города</li> </ul> <p> </p>'

    cleaned_text = generate_summary(text_raw, 10)
    print('Raw text ->', text_raw)
    print('Cleaned text ->', cleaned_text)

    # text_proc = text_clean(text_raw)
    # tokenized_text = tokenize_words(text_proc)
    # removed_stop_words = remove_stop_words(tokenized_text)
    # morphed_string = morph_analyse(removed_stop_words)
    #
    # print('Raw text ->', text_raw)
    # print('Processed text ->', tokenized_text)
    # print('Removed stop words ->', removed_stop_words)
    # print('Analyzed text ->', morphed_string)

    # print(len(tokenize_text(text_raw)))
    # print(text_raw.count('.'))
    text = '<p>Здравствуйте!</p> <p>Наша семья ищет няню для мальчика 1 годик.<br />' \
           'Мы будем рады видеть жизнерадостного и ответственного человека, ' \
           'который готов проводить с ребенком целый день и останется с нами хотябы до лета !<br />' \
           'Я надеюсь, пройдут плавно.</p>'



