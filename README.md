# Decoding the Rhythms of Emotion: Asentimental Journey through Music Genres
##### Text Mining - Data Science Degree - NOVA IMS

##
### Project Description

In the this project, our objective is to harness the power of text mining techniques to delve into the emotional essence of various music genres. This text mining endeavor encompasses two primary tasks: genre identification and sentiment analysis. The project revolves around training a machine learning model on a comprehensive dataset containing song lyrics and corresponding genres, aiming to accurately predict the genre of a song based on its lyrical content. Additionally, we embark on a sentiment analysis journey, exploring the emotional undertones within song lyrics to decipher the diverse emotional landscape across different genres. Open-ended questions guide this segment, such as discerning predominant sentiments in specific musical genres and investigating potential changes in these sentiments over time. Ultimately, our project aims to shed light on the intricate relationship between lyrics, emotions, and music genres, unraveling the profound sentiments embedded in the rhythms of diverse musical compositions.

#### Objectives:
1. Genre Identification:
- Build a classification model to predict the genre of a song based on its lyrics.
- Utilize machine learning techniques and a dataset containing song lyrics and corresponding genres.
- Evaluate and optimize model performance through a [Kaggle competition](https://www.kaggle.com/competitions/decoding-emotion-from-music), aiming to achieve the highest accuracy in genre identification.

2. Sentiment Analysis:
- Explore emotional undertones within song lyrics to decipher the emotional essence of each genre.
- Derive sentiment ratings from lyrics and investigate notable sentiment differences within and across genres.
- Formulate and answer open-ended questions, such as changes in sentiments over the years and the impact of sentiment on genre popularity.

#### Data Description
The [Genius Lyrics Dataset](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information/data) contains public user and song information from the music lyric annotation website [genius](https://genius.com/). It includes the basic information about a song: title, artists, lyrics and genre.

##### Notes:
- The original dataset has a total of 5 million songs. However, in this project, less than 5% were used.
- The data was divided a priori in two csv files: train and test datasets.
##

### Repository Description
This repository contains all the files created during the development of our project. In the following paragraphs will be a short description of how the repository is organized and what each file contains:
- [Notebook](notebook.ipynb): this file contains the project's main code and the outputs that are fundamental to our decision-making in the development of the work.
- [Utils](utils.py): this py file contains the functions needed for the main code to work, so the code is more optimized and generalizable.
- [Report](Report.pdf): this file is the academic report of the project, containing the methodology used in the work and the conclusions drawn.
- [Metadata](metadata.txt): this text file contains the metadata of all the used data in our project.
- [README](README.md): (this) file that contains the project and the repository descriptions.
##

### Possible Improvements
While the project successfully demonstrated the capability to predict song genres using natural language processing and machine learning techniques, there are several areas for potential improvement and future exploration:

1.  Expansion of Musical Genres:
- The current study focused on seven musical genres, but acknowledging that the actual number of genres exceeds 41 categories, with further subdivision into more than 300 subcategories, a more extensive dataset with additional genre tags could lead to a more comprehensive analysis and increased value.

2. Inclusion of Non-English Songs:
- A more detailed analysis could be conducted to handle songs in other non-English languages. While translating lyrics is an option, it's essential to recognize the limitations of automatic translation tools.

3. Exploration of Textual Features:
- Testing the inclusion of remaining textual features from the dataset, such as the song artist's name, could provide insights. However, this should be approached cautiously due to the potential increase in dimensionality and noise in artist representation.

4. XGBoost Classifier Optimization:
- Further exploration and optimization of the xgboost classifier could be valuable. Despite its promising results, overfitting led to its elimination. Extensive testing of parameter combinations may mitigate overfitting, but computational constraints limited the exploration.

5. Nuanced Approaches to Sentiment Analysis:
- Consideration of more nuanced approaches to sentiment analysis, including the inclusion of additional song information or exploration of different lexicons, could enhance the understanding of emotional nuances in lyrics. This avenue may provide deeper insights into the emotional tones connected to each genre.

6. Refinement of Log Ratio Analysis:
- In the current project, log ratio analysis was employed to select the most relevant words considering the genres, using the total number of words as the denominator. However, there is a potential improvement in using the total number of observations (songs) as the denominator instead. This adjustment would better account for cases where a word is repeated excessively in one song's lyrics but minimally in the remaining dataset. By normalizing with the total number of observations, the log ratio analysis could offer a more accurate reflection of a word's relevance across the entire dataset, potentially enhancing the selection of meaningful features. Further exploration of this alternative denominator could refine the feature selection process.
##
