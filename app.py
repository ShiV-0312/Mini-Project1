from flask import Flask, request, render_template
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob


app = Flask(__name__)


@app.route('/')
def my_form():
    return render_template('form.html')


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    print(text)
    return detector(text)


def detector(inputText):

    train = [
        ('Says the Annies List political group supports third-trimester abortions on demand.', 'false'),
        ('Donald Trump is against marriage equality. He wants to go back.', 'true'),
        ('Says nearly half of Oregons children are poor.', 'true'),
        ('State revenue projections have missed the mark month after month.', 'true'),
        ("In the month of January, Canada created more new jobs than we did.", 'true'),
        ('If people work and make more money, they lose more in benefits than they would earn in salary.', 'false'),
        ('Originally, Democrats promised that if you liked your health care plan, you could keep it. One year later we know that you need a waiver to keep your plan.', 'false'),
        ("We spend more money on antacids than we do on politics.", 'false'),
        ('Barack Obama and Joe Biden oppose new drilling at home and oppose nuclear power.', 'false'),
        ('President Obama once said he wants everybody in America to go to college.', 'false')
    ]

    test = [
        ('Because of the steps we took, there are about 2 million Americans working right now who would otherwise be unemployed.', 'true'),
        ('Scientists project that the Arctic will be ice-free in the summer of 2018', 'false'),
        ("You cannot build a little guy up by tearing a big guy down -- Abraham Lincoln said it.", 'false'),
        ("One man opposed a flawed strategy in Iraq. One man had the courage to call for change. One man didn't play politics with the truth.", 'true'),
        ('When I was governor, not only did test scores improve we also narrowed the achievement gap.', 'true'),
        ("Ukraine was a nuclear-armed state. They gave away their nuclear arms with the understanding that we would protect them.", 'false')
    ]

    print("Welcome to Fake App !")
    print("_____________________")
    print(" Training ...")

    cl = NaiveBayesClassifier(train)
    print(" Testing ...")

    print("your test accuracy is ", cl.accuracy(test))
    print("_____________________")


    classified_text = cl.classify(inputText)
    return classified_text
