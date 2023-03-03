import re
from typing import Set
from transformers import GPT2TokenizerFast
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk
import openai
import pickle
import tiktoken
from typing import List, Dict
from PyPDF2 import PdfReader
import PyPDF2
import re
import os
import spacy
import io
import docx2txt 
import fitz 
import subprocess
import pdf2docx
import shutil
import question
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"



##Preprocessing the Data - converts all files in 'trainingDataPdfs' to docxs and stores them in 'trainingDataDocs'
def preprocess():
    processed = []
    # Set the directory path
    dir_path = 'trainingDataDocs'
    # Get a list of all files in the directory
    for root, dirs, files in os.walk(dir_path):
        # Iterate over all files within the current directory
        for file in files:
            fname = os.path.basename(os.path.join(root, file))
            if fname != '.DS_Store':
            # Record all file names in this directory
                processed.append(fname)



    # Set the directory path
    dir_path = 'trainingDataPdfs'
    # Get a list of all files in the directory
    for root, dirs, files in os.walk(dir_path):
        # Iterate over all files within the current directory
        for file in files:
            fname = os.path.basename(os.path.join(root, file))
            #Convert pdfs to docx and store in 'trainingDataDocs'
            if fname[-4::]=='.pdf':
                altered = fname[0:-4]+'.docx'
                if altered not in processed:
                    pdfToText(fname, root)
            #Copy docx into 'trainingDataDocs'
            elif fname[-5::]=='.docx' or fname[-4::]=='.doc':
                original_file_path = root+'/'+fname
                new_file_path = 'trainingDataDocs/'+fname
                shutil.copyfile(original_file_path, new_file_path)


# Converts pdfs to docxs and stores in 'trainingDataDocs'
def pdfToText(fileName, root):

    pdf_path = root+'/'+fileName

    docx_path = 'trainingDataDocs/'+fileName[0:-4]+'.docx'

    pdf2docx.parse(pdf_path, docx_path)
    

# From all files in 'trainingDataDocs', create training data
def extractData(fileName):

    # Set Title of data to be aquired
    title = fileName

    docx_path = 'trainingDataDocs/'+fileName

    # Extract text from docxs files
    fullText = docx2txt.process(docx_path)
    
    # Determine section headers
    keywords = ['abstract', 'introduction', 'experiment', 'results', 'discussion', 'conclusion', 'concluding remarks', 'references', 'acknowledgement', 'conflict of interest', 'keywords']

    lowerFullText = fullText.lower()

    # Separate text into (keyword, content) pairs
    pattern = '|'.join(map(re.escape, keywords))
    sections = [(kw, section.strip()) for kw, section in re.findall(f'({pattern})(.*?)(?={pattern}|$)', lowerFullText, re.DOTALL)]

    # Remove content sections with useless keywords
    sections = [sec for sec in sections if (sec[0] != 'references' and sec[0] != 'acknowledgement' and sec[0] != 'conflict of interest' and sec[0] != 'keywords')]

    # Clean up data
    newSections = []
    for sec in sections:
        tokenCount = count_tokens(sec[1])
        if tokenCount > 1500:
            reduced = reduce_long(sec[1], 1500)
            tokenCount = count_tokens(reduced)
        else:
            reduced = sec[1]

        reduced = reduced.replace("\n", "")
        reduced = reduced.replace("\t", "")
        reduced = re.sub(' +', ' ', reduced)

        # (keyword, content) -> (title, keyword, content, tokenCount)
        newTup = (title, sec[0], reduced, tokenCount)
        newSections.append(newTup)

    # Return [(title, keyword, content, tokenCount), ... (title, keyword, content, tokenCount)]
    return newSections


# Count the number of tokens in a string
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Reduce a long text to a maximum of `max_len` tokens by potentially cutting at a sentence end
def reduce_long(
    long_text: str, long_text_tokens: bool = False, max_len: int = 590
) -> str:

    if not long_text_tokens:
        long_text_tokens = count_tokens(long_text)
    if long_text_tokens > max_len:
        sentences = sent_tokenize(long_text.replace("\n", " "))
        ntokens = 0
        for i, sentence in enumerate(sentences):
            ntokens += 1 + count_tokens(sentence)
            if ntokens > max_len:
                return ". ".join(sentences[:i][:-1]) + "."

    return long_text


##Embedddings

def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> List[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df: pd.DataFrame) -> Dict[tuple[str, str], List[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.content) for idx, r in df.iterrows()
    }

def load_embeddings(fname: str) -> dict[tuple[str, str], List[float]]:
    """
    Read the document embeddings and their keys from a CSV.
    
    fname is the path to a CSV with exactly these named columns: 
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """
    
    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "title" and c != "heading"])
    return {
           (r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }


def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

# def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> List[(float, (str, str))]:
def order_document_sections_by_query_similarity(query: str, contexts: Dict[tuple[str, str], np.ndarray]) -> List[tuple[float, tuple[str, str]]]:

    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
        Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities



def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
    
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"
    
def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n")



#Run the code

preprocess()

allData = []
dir_path = 'trainingDataDocs'
# Get a list of all files in the directory
for root, dirs, files in os.walk(dir_path):
    # Iterate over all files within their current directory
    for file in files:
        if file != '.DS_Store':
            allData+=extractData(file)


df = pd.DataFrame(allData, columns=["title", "heading", "content", "tokens"])
df = df[df.tokens>40]
# df = df.drop_duplicates(['title','heading'])
df = df.reset_index().drop('index',axis=1) # reset index
df.head()
df.to_csv('trainingData.csv', index=False)

document_embeddings = compute_doc_embeddings(df)

myQuestion = "What are the difficult challenges to downscaling dynamic random access memory?"


order_document_sections_by_query_similarity(myQuestion, document_embeddings)[:5]

MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

f"Context separator contains {separator_len} tokens"




prompt = construct_prompt(
    myQuestion,
    document_embeddings,
    df
)

print(  "===\n", prompt)

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}



query = myQuestion
answer = answer_query_with_context(query, df, document_embeddings)
print(answer)