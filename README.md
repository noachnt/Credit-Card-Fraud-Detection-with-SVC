# Credit Card Fraud Detection with SVC
This program detect fraud based on features.

## Team Member
- [Liang Cai](https://github.com/acailuv)
- [Noach Nathanael T.](https://github.com/noachnt)
- Calvin Joe

## Requirement
- Python Version 3.6 or higher. [(Download Python here)](https://www.python.org/downloads/)
	- Python 3.6 or higher should include pip if you correctly tick the checkbox that says "Add to PATH".
	- Refer to the tutorial below if could not install libraries through pip even though you have 	installed Python.
- Install pip. [(Tutorial how to Install pip)](https://www.makeuseof.com/tag/install-pip-for-python/)
- Install dash. 
	- Open your command prompt or console.
	- Type `pip install dash`.
	- Wait until finished.
- Install pandas.
	- Open your command prompt or console.
	- Type `pip install pandas`.
	- Wait until finished.
- Install plotly.
	- Open your command prompt or console.
	- Type `pip install plotly`.
	- Wait until finished.
- Install sklearn.
	- Open your command prompt or console.
	- Type `pip install sklearn`.
	- Wait until finished.
- Download dataset from Kaggle. [(Download dataset here)](https://www.kaggle.com/mlg-ulb/creditcardfraud/downloads/creditcardfraud.zip/3)
	- You might have to log in first. 
	- Extract the zip file in the same folder as the python script. (`fraud_detection_svc.py`)
	
## Setting Up 
In this section you will be guided through the process of setting up dash application. All you need to do is double-click `fraud_detection_svc.py`. The process might take some time so be patient. This message will appear from the console when the preprocessing is done and the server is up:\
![alt text](https://i.imgur.com/XyXQ8B4.png "Server already set up")\
You can now open your browser and browse `http://127.0.0.1:8050/` to see the app in action.

## The Application
When you open up the app, this page will show up by default:\
![alt text](https://i.imgur.com/wYGnmy5.png "Application default page")\
There are three tabs at the top which will show you three graph categories which is: Statistics; where you will find 2 graphs: One will show you a pie chart of the comparison between the amount of legit transactions and the amount of fraud transactions. The second one will show the confusion matrix of the prediction done by this application (powered by SVC or Support Vector Classifier).\
![alt text](https://i.imgur.com/XP9Qx6j.png "Confusion Matrix and Prediction Accuracy")\
Confusion matrix will show the errors (alongside the correct) guesses that are done by the machine. The text below the confusion matrix is the accuracy of the prediction with current parameters (this will be explained later on). Going to the 'Heatmap' tab, this is what you will see when you first click the tab:\
![alt text](https://i.imgur.com/VXSsERZ.png "'Heatmap' tab page")\
This page shows you the correlation between features. There are positive point (>0) and negative points (\<0) of correlation. If the z-value of that block is less than 0 it means that the correlation is 'backwards', and if it is higher than 0 it means that the correlation is 'forward'. For instance, consider this picture:\
![alt text](https://i.imgur.com/U04tryO.png)\
the feature 'V2' has a z-value of 0.09128865 which is larger than 0. This means that if 'V2' got higher, the 'Class' feature also got higher. This is a 'forward' correlation. Take a look at this next case:\
![alt text](https://i.imgur.com/GEYJgEm.png)\
The feature 'V17' has a z-value of -0.3264811 which is less than 0. This means that if 'V17' got higher, the 'Class' feature will contradict the trend. Ergo, 'Class' feature will get lower as 'V17' feature get higher.\
The third will show you sliders as input for the features. This is what you see when you click the third tab:\
![alt text](https://i.imgur.com/xUjf2q1.png)\
By default, the silders' value will be set to zero. However, you can play around with the value and click the 'Predict' Button to the upper right to command the machine to predict whether that transaction (with its assigned values) is a fraudulent transaction or not.

## The Script
This section will explain the script itself. What libraries are used, how the machine come up with a verdict, and how the script projects the data in form of graph via Dash.
### Part One: Essential Libraries
![alt text](https://i.imgur.com/rPabp5r.png)\
This is all the libraries needed for this script to run. The script is already well-explained. However, we will go through what each library does in general. First off: `dash`, `dash_core_components`, and `dash_html_component`. `dash` is used to create web-based application. `dash_core_components` is used to create graphs within the application. `dash_html_components` is used as a source of html tags like `<p>`, `<div>`, and many other things. The tags are called like this: `html.P()` in the script. Where html is the 'alias' of the `dash_html_components`. Because the library is imported like this: `import dash_html_components as html`, `html` is what we meant by 'alias'.\
Next up is `datetime`. As the name suggest, this is used to make date and time operations. Especially converting from seconds to hour and extracting the hour from the full timestamp (`1970-01-01 23:59:59 to 23`).\
`pandas` library is used to handle data frames. `plotly` is basically used to graph objects. `numpy` and `math` are used to execute mathematical operations. Finally, `sklearn` is used to do machine learning (`from sklearn import svm`) and confusion matrix (`from sklearn.metrics import confusion_matrix`).
### Part Two: How It Works
![alt text](https://i.imgur.com/BMOODQi.png)\
First we prepare the data frame with creditcard.csv as its source (download link in the requirement above). We add new column called 'Datetime; to the drataframe to hold time data in this format: `1970-01-01 23:59:59` yy/mm/dd is default because we only convert the 'time'. After that we add another column called 'hour; to the data frame to hold 'hour part of the 'Datetime' in this format `1970-01-01 23:59:59` -> `23`. Because we only need hour column from now on, we drop 'Time' and 'Datetime' columns. Next we make a new data frame called `legit` and `fraud` to hold legit and fraud daata transaction respectively. After we make the data frame we make random sampling  from the legit transaction and we make training data. Inside training data there are mix fraud with legit transaction to balance it out.
![alt text](https://i.imgur.com/RSZevDb.png)\
Second we make training data called `x_train` and drop `class` since we dont need it in the training data. After that we make training data called `y_train` containing the class column. We drop the `class` in `x_train` because the `class` contains the verdict like its a fraud or not a fraud. So we make that the `x_train` hold the `pattern` or `data`. The one that going to verdict this is a fraud or no is `y_train` that contain the `class`. After that we make a new SVC (Support Vector Classifier) object with linear kernel and we command the computer to study the training data with `.fit()`. Next we predict all the data in the data frame to be used in the confusion matrix and generate the confusion matrix values to be drawn in graph. After that we style the sheet and make a new dash object with `external_stylesheets` and suppress exception in oreder to work with tabs.
### Part Three: How It Displays
![alt text](https://i.imgur.com/XU2BQHl.png)\
We make the GUI (Graphical User Interface) using html. First we make the title and then we generate the tabs that includes: statistic, heatmap and predict tab. Then we callback input and output. The output to `tab-content` with the input from `maintab	`. 
![alt text](https://i.imgur.com/E8J5xDl.png)\
Code above all we do is make a function that reacts with the input and ouput.
![alt text](https://i.imgur.com/3PDtNAT.png)\
After that we make slider to input the value of `V1` - `V28` (features), and we make the predict button then make element to display feature details and the result.\
![alt text](https://i.imgur.com/d7vRSRA.png)\
Next, we make a function to update the output and to give an output if its a fraud or not.
![alt text](https://i.imgur.com/Vb0QIos.png)\
The last thing is we make `live updater` for each of the features from `V1` - `V28`. These callback function is used to update the details of each slider mentioned above.


## Closing
Machine learning is this fascinating thing that can make computers study huge data and make them predict a verdict (at least in this case). We have some trouble in the beginning but we managed to make it through. It is as they say: "If there is a will, there is a way". Thank you very much.
