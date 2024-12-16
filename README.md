# temporal-cultural-norms
This is the code used to scrape subtitle data from Hollywood and Bollywood movies and parse through subtitle contexts for explicit and random (matching vs. non-matching) mentions of **shame** and **pride**-related keywords. We also included a `.ipynb` file that persforms Agglomerative using SBERT on the processed data to form clusters and social norms.

### Setting up
Make sure to install all necessary python libaries and packages.
Stay in the root folder, and make sure to create the following directories:
- input/
- parsed_input/
- gpt_data/
- processed_data/

### Files
- parser_code/bollwood_parser.py
- parser_code/hollwood_parser.py
- README.md
- bollywood_scraper.py
- gpt.py
- processing.py

### Data
*Access data in this [Google Drive link](https://drive.google.com/drive/folders/1XzEAzx93VEOT8FtzhzEzXwCEQihiwLVc?ths=true).*

Once you have downloaded all the data locally, please upload them to your project using the same directory structure as the one in Drive.
Below is a detailed explanation of what the data folder includes.

#### Input (not included, check details below)
*RAW DATA*

- Hollywood dataset:
    - Download and unzip `meta` and `subtitles` data 
    - Upload them inside the folder `input/hollwood/`
- Bollywood scraped data:
    - Download and unzip `bollywood_sub` inside the `input/bollywood/` folder on Drive.
    - Upload the unzipped folder into the folder `input/bollywood/` in your project and make sure the names match with the directory in `parser_code/bollywood_parser.py`
* Refer to the `parser_code/` folder to clean and format the raw data to be ready for the next steps.

*PARSED DATA*

This includes the **matching context** results from `parser_code` files.
**Note that these can be downloaded from DRIVE or manually ran**
- Hollywood matching contexts:
    - Download from Drive (`matching_hollywood.csv` and `random_hollywood.csv`) or run it manually using the command  `python3 parser_code/hollywood_parser.py`
    - Upload it inside the folder `parsed_input`
- Bollywood scraped data:
    - Download from Drive (`matching_bollywood.csv` and `random_bollywood.csv`) or run it manually using the command  `python3 parser_code/bollywood_parser.py`
    - Upload it inside the folder `parsed_input`

*GPT DATA*

This includes data from calling GPT-4o on data from *PARSED DATA*.

**Note that these can be downloaded from DRIVE or manually ran `python3 gpt.py`**
- Hollywood matching contexts:
    - Download from Drive (`entire_hollywood_gpt-4o.csv`) or run it manually using the command  `python3 gpt.py`
    - Upload it inside the folder `gpt_data`
- Bollywood scraped data:
    - Download from Drive (`entire_bollywood_gpt-4o.csv`) or run it manually using the command  `python3 gpt.py`
    - Upload it inside the folder `gpt_data`

#### Output
This includes data from running **Sentence Embeddings (SBERT)** and **Agglomerative Clustering** on the *GPT Data*.
This includes several different files that range from the word embeddings (without duplicates), norm to cluster mappings, and the final results as well (that can be read in the notebook to create visualizations).

Please cross-check with the `.ipynb` notebook: `processing_ipynb` to see what each of the file means. 

### How to run
*Assuming you don't have access to Google Drive Folder*
1. Language and libraries. Make sure you have `python3` and all necessary libraries installed.
2. Obtain raw data
* Obtain Hollywood data from the Google Drive folder (or cross-check with the [Kaggle dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=ratings.csv)).
* Obtain Bollywood data by running `python3 bollywood_scraper.py`
3. Parse through raw data to obtain `matching` vs `random` contexts
* Obtain Hollywood contexts by running `python3 parser_code/hollywood_parser.py`
* Obtain Bollywood contexts by running `python3 parser_code/bollywood_parser.py`
This will output the `matching_{cinema}.csv` and `random_{cinema}.csv` files inside `parsed_input/` folder.
3. Call GPT-4o calls on the parsed data
* Obtain and replace the OpenAI key with yours
* Make sure you configure the directory to point to the right parsed Hollywood/Bollywood context files
* Run `python3 gpt.py`
4. Run either the python file or notebook file to process, analyze, and perform clustering on your results from part (3) (`processing.ipynb` or `processing.py`)
* Load the python notebook and run each cell. Make sure you configure the files and encoding and duplicate booleans as needed

### Citations
Please leave us a star and cite our paper(s) if you find our work helpful.
```
Citation to add in Github link - @misc{rai2024socialnormscinemacrosscultural,
      title={Social Norms in Cinema: A Cross-Cultural Analysis of Shame, Pride and Prejudice},
      author={Sunny Rai and Khushang Jilesh Zaveri and Shreya Havaldar and Soumna Nema and Lyle Ungar and Sharath Chandra Guntuku},
      year={2024},
      eprint={2402.11333},
      archivePrefix={arXiv},
      primaryClass={cs.CY},
      url={https://arxiv.org/abs/2402.11333},
}
```




