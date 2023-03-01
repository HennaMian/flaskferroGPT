How it Works:

This program converts all .pdf and .docx files in the "trainingDataPdfs" directory to .docx files and store them in the "trainingDataDocs" directory. It then trains CHATGPT on all .docx files in the "trainingDataDocs" directory.

You can run the testing.py file in order to ask the model a question.

Note: feel free to delete files in the "trainingDataPdfs" directory after they have been converted and stored in "trainingDataDocs". They are no longer needed. However, not delete the "trainingDataPdfs" directory itself.


How to Start:

In the terminal, navigate to the "training" folder 

Install requirements by running "pip install -r requirements.txt"

Uncomment nltk.download('punkt') on line 4 of the "question.py" file for your first run.

Go to https://platform.openai.com/overview , make an account, select "Personal" in the top right corner, then select "View API keys"

Generate an open-ai api key and copy it. It will disappear after viewing it once.

Replace "YOUR_KEY_HERE" on line 6 of "question.py" with your API key.



How to Train and Ask a Question:

Fill "trainingDataPdfs" folder with .pdf and .docx files to train ChatGPT on

Replace "What is your question?" on line 7 of "question.py" with your question.

In terminal, run "python testing.py"

Feel free to comment "nltk.download('punkt')" on line 4 of "question.py" again. Feel free to delete files in "trainingDataPdfs" directory.