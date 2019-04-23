# Film-Analysis flask server

This is a REST API for the back-end of the film-analysis web app. It is written in python that processes queries and returns the most similar movies and classifies movies into genre based on their summary.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

1) Python 3
2) Windows
3) Git
4) Heroku


### Installing (For Windows Users)

A step by step series of examples that tell you how to get a development env running

1) cd to the directory where requirements.txt is located

```
cd .\flask_server\
```

2) activate your virtualenv 

```
venv\Scripts\activate
```

3) Install packages from requirements.txt

```
pip install -r requirements.txt
```


## Running the code

To run the flask server locally
```
py .\api.py
```

## Deployment

To deploy the application to Heroku after setting up heroku cli on your machine using git
```
git push heroku master
```

* To set up Heroku cli on your machine (https://devcenter.heroku.com/articles/git)

## Built With

* [FLASK](http://flask.pocoo.org/docs/1.0/) - The web framework used
* [Python 3](https://docs.python.org/3/) - Programming Language Used
* [PANDAS](https://pandas.pydata.org/) - Data Analysis Library Used
* [NLTK](http://www.nltk.org/index.html) - Text Analysis Library Used
* [NumPy](https://www.numpy.org/) - Scientific Computing Library Used
* [Heroku](https://devcenter.heroku.com) - Deployment Used


## Authors

* **Jose Cervantes** 


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Textbook 'Introduction to Information Retrieval' by Christopher D. Manning, Hinrich Schütze, and Prabhakar Raghavan (https://nlp.stanford.edu/IR-book/information-retrieval-book.html)
* Dr. Park's Template Code (https://drive.google.com/drive/folders/1z9zSSVeo7GvXjuoiF6_h-Ogv41e4PCzP?usp=sharing)
* Minimal Flask Server (http://flask.pocoo.org/docs/0.12/quickstart/#a-minimal-application)