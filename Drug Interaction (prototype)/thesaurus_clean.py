# imports
import pandas as pd
import time
import PyPDF2
import pdftotext
import tqdm
import py2neo
import os

my_utility_file = './thesaurus_interactions.pdf'

conclusion_type = ['''Association DECONSEILLEE''',
                   '''Précaution d'emploi''',
                   '''A prendre en compte''',
                   '''CI''',
                   '''ASDEC''',
                   '''APEC''',
                   '''PE''',
                   '''ASDEC - APEC''',
                   '''CI - ASDEC - APEC''',
                   '''ASDEC - PE''',
                   '''CI - PE''',
                   '''CONTRE-INDICATION'''
                   ]

def utility_get_pages_nbr(my_file, display=False):
    '''
    function which gives the text nb of pages
    :param my_file:
    :param display:
    :return:
    '''

    pdfFileObject = open(my_file, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObject)
    count = pdfReader.numPages
    res = count
    if display is True:
        print(" the number of pages is : {0}".format(count))
    return(res)


def simple_parser(my_file):

    '''
    function which parses the text without cleaning
    :param my_file:
    :return:
    '''

    # Parse pdf automatically with package
    pdfFileObject = open(my_file, 'rb')
    pdf = pdftotext.PDF(pdfFileObject)

    # Iterate over all the pages
    my_str = ""
    for page in tqdm.tqdm(pdf):
        my_str += page

    # Get the list of all drugs
    list_of_interactions = my_str.split("\n")
    list_of_meds = [el.replace("+ ", "") for el in list_of_interactions if "+" in el]

    # Get the list of all drugs interactions
    list_of_interactions = my_str.split("\n")

    return(list_of_meds, list_of_interactions)


def clean(list_of_interactions, n_pages):
    '''
    function which cleans the princeps and its interactions organized as a single list
    :param list_of_interactions:
    :param n_pages:
    :return: inter: interactions organized as a list
    '''

    inter = []
    # to get rid of  the page numbers
    strings_int = [str(el) for el in range(n_pages - 1)]

    # to remove the extra spaces & return all the drugs and their associated interactions (+ sign)
    # in a  single list
    for el in list_of_interactions:
        el = el.strip()
        if el.upper() == el and el not in strings_int:
            inter.append(el)
    return (inter)


def parser2(list_of_interactions):
    '''
    function which fixes the raw list of interactions when text is written over 2 lines
    :param list_of_interactions:
    :return: list_of_interactions:
    '''

    for ix, el in enumerate(list_of_interactions):
        if ix > 0:
            if list_of_interactions[ix].strip() == 'ADSORBANTS':
                list_of_interactions[ix - 1] = list_of_interactions[ix - 1] + " " + 'ADSORBANTS'
                list_of_interactions.pop(ix)
    return list_of_interactions


def parser3(list_of_interactions):
    '''
    function which returns drugs, interactions and recommendations from the raw list of interactions
    :param list_of_interactions:
    :return: df_data_med_inter_poso:
    '''

    data_med_inter_poso = []
    #print(list_of_interactions)

    for i in range(len(list_of_interactions)):
        posoloj = ""
        if list_of_interactions[i].upper() == list_of_interactions[i] and "     " not in list_of_interactions[i]:
            if not list_of_interactions[i].strip().startswith("+"):
                keyz = list_of_interactions[i].strip()

            else:
                interacted = list_of_interactions[i].strip()

                counter_text = 1
                while True:
                    if "     " in list_of_interactions[i + counter_text] or \
                            " Il convient de prendre" in list_of_interactions[i + counter_text]:

                        posoloj += list_of_interactions[i + counter_text]
                        counter_text += 1
                    else:
                        break
                data_med_inter_poso.append([keyz, interacted.replace("+", ""), posoloj])

    df_data_med_inter_poso = pd.DataFrame(data_med_inter_poso)
    df_data_med_inter_poso.columns = ["princeps", "interacted_with", "conclusion"]
    return df_data_med_inter_poso


def pprepare(df_data_med_inter_poso, conclusion_type):
    '''
    function which creates a  summarized column (conclusion type) from the posology/recommandation text
    :param data_med_inter_poso:
    :param conclusion_type:
    :return df_data_poso_ccl:
    '''

    df = df_data_med_inter_poso

    # line fixing - line ill formatted
    results = []
    #print(df)
    for i in range(df.shape[0]):
        # remove the arrow character "\x1a"
        problems = (" ").join([el.strip() for el in df.iloc[i:i + 1]["conclusion"].values[0].split("  ") if
                               el.strip() not in conclusion_type and el != '']).replace("\x1a", "")
        ccl = ("").join([el.strip() for el in df.iloc[i:i + 1]["conclusion"].values[0].split("  ") if
                         el.strip() in conclusion_type and el != ''])

        if ccl == '':
            for el in conclusion_type:
                if el in problems:
                    ccl += el
        results.append([problems, ccl])

    df_data_poso_ccl = pd.DataFrame(results)
    #print(df_data_poso_ccl)
    df_data_poso_ccl.columns = ["text","verdict"]
    return df_data_poso_ccl


def get_merged_dataset(df_data_med_inter_poso, df_data_poso_ccl,save=True):
    '''
    function which outputs the master dataframe
    :param df_data_med_inter_poso:
    :param df_data_poso_ccl:
    :return:
    '''

    result = pd.concat([df_data_med_inter_poso, df_data_poso_ccl], axis=1)
    result_subset = result[["princeps", "interacted_with", "verdict"]]
    result_subset["princeps"] = result_subset["princeps"].apply(lambda x:x.strip())
    result_subset["interacted_with"] = result_subset["interacted_with"].apply(lambda x:x.strip())
    if save is True:
        result_subset.to_csv("./thesaurus_cleaned.csv", sep=',', index=None)

    return result_subset


if __name__ == "__main__":
    _, interactions = simple_parser(my_utility_file)

    #print(list_of_interactions)
    n_pages = utility_get_pages_nbr(my_utility_file, display=False)

    #interactions = clean(list_of_interactions, n_pages)

    interactions_step_II = parser2(interactions)
    interactions_step_III = parser3(interactions_step_II)
    interactions_step_IV = pprepare(interactions_step_III, conclusion_type)
    get_merged_dataset(interactions_step_III, interactions_step_IV)