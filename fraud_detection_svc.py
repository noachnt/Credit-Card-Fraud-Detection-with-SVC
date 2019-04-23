### importing essetial libraries
# dash (needed to be installed via pip or the like)
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

# datetime
import datetime

# pandas (needed to be installed via pip or the like)
import pandas as pd

# plotly (needed to be installed via pip or the like)
import plotly.graph_objs as go
import plotly.plotly as py

# numpy (needed to be installed via pip or the like)
import numpy as np

# math
import math

# sklearn (needed to be installed via pip or the like)
from sklearn.metrics import confusion_matrix
from sklearn import svm

def convert(seconds):
    return datetime.datetime.utcfromtimestamp(seconds)

### Do all data preprocessing here
df = pd.read_csv("creditcard.csv", engine='python') # prepare data frame with creditcard.csv as its source
df['Datetime'] = df.Time.apply(convert) # add new column 'Datetime' to the dataframe to hold time data in this format: 1970-01-01 23:59:59
df['hour'] = df.Datetime.dt.hour # add new column 'hour' to the data frame to hold the 'hour' part of 'DateTime': 1970-01-01 23:59:59 -> 23
df = df.drop(['Time', 'Datetime'], axis=1) # drop 'Time' and 'Datetime' columns, since we only need 'hour' column from now on

legit = df[df.Class == 0] # make a new data frame called 'legit' to hold data of legit transactions
fraud = df[df.Class == 1] # make a new data frame called 'fraud' to hold data of fraud transactions

df_sample = legit.sample(300) # random sampling from the legit transactions
df_train = fraud[:300].append(df_sample) # mix fraud with legit transactions to balance out the training data

x_train = np.asarray(df_train.drop(['Class'], axis=1)) # make training data 'x_train' (drop 'class' column since we don't need it in this training data)
y_train = np.asarray(df_train['Class']) # make training data 'y_train' contaitning the 'class' column

classifier = svm.SVC(kernel='linear') # make a new SVC object with 'linear' kernel
classifier.fit(x_train, y_train) # command the computer to study the data with .fit()

predict_all = classifier.predict(df.drop(['Class'], axis=1)) # this data predict all the data in the data frame (to be used in confusion matrix)
cm = confusion_matrix(df['Class'], predict_all) # this generate the confusion matrix values (to be drawn in graph)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css'] # sheet styling css

app = dash.Dash(external_stylesheets=external_stylesheets) # make a new dash object with 'external_stylesheets' as its styling
app.config['suppress_callback_exceptions'] = True # suppress exceptions in order to work with tabs

### contents of the web app
app.layout = html.Div(children=[
    html.H2('Identifying Credit Card Fraud'), # the big old title at the top
    dcc.Tabs(id="maintab", value="tab-1", children=[ # this generates the tabs
        dcc.Tab(label="Statistics", value='tab-1'), # statistics tab
        dcc.Tab(label="Heatmap", value='tab-2'), # heatmap tab
        dcc.Tab(label="Predict", value='tab-3') # predict tab
    ]),
    html.Div(id='tab-content') # make a 'Div' html element to contain the content of those already-generated tabs
])

@app.callback( # 'callback' i/o
    Output('tab-content', 'children'), # outputs to 'tab-content'
    [Input('maintab', 'value')] # with the input from 'maintab'
)
def render_content(tab): # 'callback' function that reacts with the i/o mentioned above.
    if tab == 'tab-1': # if 'tab-1' or statistics tab is clicked
        return html.Div([ # return a 'Div' element which contains:
            dcc.Graph( # 1> a graph
                figure=go.Figure( # which is a figure
                    data=[
                        go.Pie( # represented as a pie chart
                        labels = ['Legit','Fraud'], # contains data which is called 'Legit' and 'Fraud'
                        values = [legit.V1.count(), fraud.V1.count()] # fill the values with the count of rows (actualized by counting V1 columns in legit and fraud data frame)
                        )
                    ],
                    layout=go.Layout( # change some layout properties to this graph (pie chart)
                        title='Legit vs Fraud transaction frequency', # set the graph's title
                        height = 600, # change the graph's height
                        width = 600 # change the graph's width
                    )
                )
            ),
            dcc.Graph( # 2> a graph
                figure = go.Figure( # which is a figure
                    data=[
                        go.Heatmap(z = cm, # represented as a heatmap; with the z value to 'cm' variable mentioned above
                                   x = ["Predict not fraud", "Predict fraud"], # x-axis labels
                                   y = ["Actual not fraud", "Actual fraud"] ) # y-axis labels
                    ],
                    layout = go.Layout( # change some layout properties to this graph (heatmap)
                        title='Confusion matrix for SVM', # set the graph's title
                        height=600, # change the graph's height
                        width=600 # change the graph's width
                    )
                )
            ),
            html.P("Accuracy is " + str((cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1])) * 100) + "%") # the accuracy of the prediction, based on the confusion matrix
            # formula: (true_predictions) / (all_data_count) * 100
        ])
    elif tab == 'tab-2': # else if 'tab-2' or heatmap tab is clicked
        return html.Div([ # return a 'Div' element which contains:
            dcc.Graph( # 1> a graph
                figure=go.Figure( # which is a figure
                    data=[
                        go.Heatmap( # represented as a heatmap
                            x=df.columns, # x-axis labels
                            y=df.columns, # y-axis labels
                            z=df.corr(method='pearson') # assign z-axis value to be a correlation table
                        )
                    ],
                    layout=go.Layout( # change some layout properties to this graph (heatmap) 
                        title = 'Correlation between variables', # set the graph's title
                        height = 800, # change the graph's height
                        width = 800 # change the graph's width
                    )
                )
            )
        ])
    elif tab == 'tab-3': # else if 'tab-2' or heatmap tab is clicked
        return html.Div([ # return a 'Div' element which contains:
            html.Div([ # a 'Div' element which contains:
                dcc.Slider(id='SV1', min=math.floor(df.V1.min()), max=math.ceil(df.V1.max()), step=0.1, value=0), # slider for V1
                html.Div(id='TV1'), # container for V1 slider's details
                dcc.Slider(id='SV2', min=math.floor(df.V2.min()), max=math.ceil(df.V2.max()), step=0.1, value=0), # slider for V2
                html.Div(id='TV2'), # container for V2 slider's details
                dcc.Slider(id='SV3', min=math.floor(df.V3.min()), max=math.ceil(df.V3.max()), step=0.1, value=0), # slider for V3
                html.Div(id='TV3'), # container for V3 slider's details
                dcc.Slider(id='SV4', min=math.floor(df.V4.min()), max=math.ceil(df.V4.max()), step=0.1, value=0), # slider for V4
                html.Div(id='TV4'), # container for V4 slider's details
                dcc.Slider(id='SV5', min=math.floor(df.V5.min()), max=math.ceil(df.V5.max()), step=0.1, value=0), # slider for V5
                html.Div(id='TV5'), # container for V5 slider's details
                dcc.Slider(id='SV6', min=math.floor(df.V6.min()), max=math.ceil(df.V6.max()), step=0.1, value=0), # slider for V6
                html.Div(id='TV6'), # container for V6 slider's details
                dcc.Slider(id='SV7', min=math.floor(df.V7.min()), max=math.ceil(df.V7.max()), step=0.1, value=0), # slider for V7
                html.Div(id='TV7'), # container for V7 slider's details
                dcc.Slider(id='SV8', min=math.floor(df.V8.min()), max=math.ceil(df.V8.max()), step=0.1, value=0), # slider for V8
                html.Div(id='TV8'), # container for V8 slider's details
                dcc.Slider(id='SV9', min=math.floor(df.V9.min()), max=math.ceil(df.V9.max()), step=0.1, value=0), # slider for V9
                html.Div(id='TV9'), # container for V9 slider's details
                dcc.Slider(id='SV10', min=math.floor(df.V10.min()), max=math.ceil(df.V10.max()), step=0.1, value=0), # slider for V10
                html.Div(id='TV10'), # container for V10 slider's details
                dcc.Slider(id='SV11', min=math.floor(df.V11.min()), max=math.ceil(df.V11.max()), step=0.1, value=0), # slider for V11
                html.Div(id='TV11'), # container for V11 slider's details
                dcc.Slider(id='SV12', min=math.floor(df.V12.min()), max=math.ceil(df.V12.max()), step=0.1, value=0), # slider for V12
                html.Div(id='TV12'), # container for V12 slider's details
                dcc.Slider(id='SV13', min=math.floor(df.V13.min()), max=math.ceil(df.V13.max()), step=0.1, value=0), # slider for V13
                html.Div(id='TV13'), # container for V13 slider's details
                dcc.Slider(id='SV14', min=math.floor(df.V14.min()), max=math.ceil(df.V14.max()), step=0.1, value=0), # slider for V14
                html.Div(id='TV14'), # container for V14 slider's details
                dcc.Slider(id='SV15', min=math.floor(df.V15.min()), max=math.ceil(df.V15.max()), step=0.1, value=0), # slider for V15
                html.Div(id='TV15'), # container for V15 slider's details
                dcc.Slider(id='SV16', min=math.floor(df.V16.min()), max=math.ceil(df.V16.max()), step=0.1, value=0), # slider for V16
                html.Div(id='TV16'), # container for V16 slider's details
                dcc.Slider(id='SV17', min=math.floor(df.V17.min()), max=math.ceil(df.V17.max()), step=0.1, value=0), # slider for V17
                html.Div(id='TV17'), # container for V17 slider's details
                dcc.Slider(id='SV18', min=math.floor(df.V18.min()), max=math.ceil(df.V18.max()), step=0.1, value=0), # slider for V18
                html.Div(id='TV18'), # container for V18 slider's details
                dcc.Slider(id='SV19', min=math.floor(df.V19.min()), max=math.ceil(df.V19.max()), step=0.1, value=0), # slider for V19
                html.Div(id='TV19'), # container for V19 slider's details
                dcc.Slider(id='SV20', min=math.floor(df.V20.min()), max=math.ceil(df.V20.max()), step=0.1, value=0), # slider for V20
                html.Div(id='TV20'), # container for V20 slider's details
                dcc.Slider(id='SV21', min=math.floor(df.V21.min()), max=math.ceil(df.V21.max()), step=0.1, value=0), # slider for V21
                html.Div(id='TV21'), # container for V21 slider's details
                dcc.Slider(id='SV22', min=math.floor(df.V22.min()), max=math.ceil(df.V22.max()), step=0.1, value=0), # slider for V22
                html.Div(id='TV22'), # container for V22 slider's details
                dcc.Slider(id='SV23', min=math.floor(df.V23.min()), max=math.ceil(df.V23.max()), step=0.1, value=0), # slider for V23
                html.Div(id='TV23'), # container for V23 slider's details
                dcc.Slider(id='SV24', min=math.floor(df.V24.min()), max=math.ceil(df.V24.max()), step=0.1, value=0), # slider for V24
                html.Div(id='TV24'), # container for V24 slider's details
                dcc.Slider(id='SV25', min=math.floor(df.V25.min()), max=math.ceil(df.V25.max()), step=0.1, value=0), # slider for V25
                html.Div(id='TV25'), # container for V25 slider's details
                dcc.Slider(id='SV26', min=math.floor(df.V26.min()), max=math.ceil(df.V26.max()), step=0.1, value=0), # slider for V26
                html.Div(id='TV26'), # container for V26 slider's details
                dcc.Slider(id='SV27', min=math.floor(df.V27.min()), max=math.ceil(df.V27.max()), step=0.1, value=0), # slider for V27
                html.Div(id='TV27'), # container for V27 slider's details
                dcc.Slider(id='SV28', min=math.floor(df.V28.min()), max=math.ceil(df.V28.max()), step=0.1, value=0), # slider for V28
                html.Div(id='TV28'), # container for V28 slider's details
                dcc.Slider(id='SAmount', min=math.floor(df.Amount.min()), max=math.ceil(df.Amount.max()), step=0.1, value=1), # slider for Amount
                html.Div(id='TAmount'), # container for Amount slider's details
                dcc.Slider(id='SHour', min=math.floor(df.hour.min()), max=math.ceil(df.hour.max()), step=1, value=12), # slider for Hour or Time
                html.Div(id='THour'), # container for Hour or Time slider's details
            ], style={'float': 'left', 'width': '70%', 'margin-right': 20}), # css styling
            html.Div([ # another 'Div' element which contains:
                html.Button('Predict', id='button', n_clicks=0), # a 'Predict' button
                html.Div(id='Result') # and a 'Div' element to display feature details and the result
            ])
        ])
@app.callback( # 'callback' i/o for the sliders
    Output('Result', 'children'), # output to 'Result' element
    [Input('button', 'n_clicks')], # with these inputs
    [State('SV1', 'value'),
     State('SV2', 'value'),
     State('SV3', 'value'),
     State('SV4', 'value'),
     State('SV5', 'value'),
     State('SV6', 'value'),
     State('SV7', 'value'),
     State('SV8', 'value'),
     State('SV9', 'value'),
     State('SV10', 'value'),
     State('SV11', 'value'),
     State('SV12', 'value'),
     State('SV13', 'value'),
     State('SV14', 'value'),
     State('SV15', 'value'),
     State('SV16', 'value'),
     State('SV17', 'value'),
     State('SV18', 'value'),
     State('SV19', 'value'),
     State('SV20', 'value'),
     State('SV21', 'value'),
     State('SV22', 'value'),
     State('SV23', 'value'),
     State('SV24', 'value'),
     State('SV25', 'value'),
     State('SV26', 'value'),
     State('SV27', 'value'),
     State('SV28', 'value'),
     State('SAmount', 'value'),
     State('SHour', 'value'),
     ]
)
def update_output(*args): # 'callback' function that reacts with the i/o mentioned above. 
    values = list(args)
    cols = list(df.drop(['Class'], axis=1).columns)
    values.pop(0)
    zipp = list(zip(cols, values))
    dff = pd.DataFrame([values], columns = cols)
    result = classifier.predict(dff)
    if result[0] == 0:
        out = "The transactions with parameters " + str(zipp) + "is not a fraudulent transaction"
    else:
        out = "The transactions with parameters " + str(zipp) + "is a fraudulent transaction"
    return html.P(out)

### Everything below is just for live updating
# these 'callback' i/o(s) and 'callback' functions is used to update the details of each sliders metioned above
@app.callback(
    Output('TV1', 'children'),
    [Input('SV1', 'value')])
def update_output(value):
    out = "V1 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV2', 'children'),
    [Input('SV2', 'value')])
def update_output(value):
    out = "V2 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV3', 'children'),
    [Input('SV3', 'value')])
def update_output(value):
    out = "V3 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV4', 'children'),
    [Input('SV4', 'value')])
def update_output(value):
    out = "V4 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV5', 'children'),
    [Input('SV5', 'value')])
def update_output(value):
    out = "V5 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV6', 'children'),
    [Input('SV6', 'value')])
def update_output(value):
    out = "V6 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV7', 'children'),
    [Input('SV7', 'value')])
def update_output(value):
    out = "V7 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV8', 'children'),
    [Input('SV8', 'value')])
def update_output(value):
    out = "V8 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV9', 'children'),
    [Input('SV9', 'value')])
def update_output(value):
    out = "V9 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV10', 'children'),
    [Input('SV10', 'value')])
def update_output(value):
    out = "V10 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV11', 'children'),
    [Input('SV11', 'value')])
def update_output(value):
    out = "V11 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV12', 'children'),
    [Input('SV12', 'value')])
def update_output(value):
    out = "V12 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV13', 'children'),
    [Input('SV13', 'value')])
def update_output(value):
    out = "V13 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV14', 'children'),
    [Input('SV14', 'value')])
def update_output(value):
    out = "V14 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV15', 'children'),
    [Input('SV15', 'value')])
def update_output(value):
    out = "V15 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV16', 'children'),
    [Input('SV16', 'value')])
def update_output(value):
    out = "V16 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV17', 'children'),
    [Input('SV17', 'value')])
def update_output(value):
    out = "V17 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV18', 'children'),
    [Input('SV18', 'value')])
def update_output(value):
    out = "V18 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV19', 'children'),
    [Input('SV19', 'value')])
def update_output(value):
    out = "V19 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV20', 'children'),
    [Input('SV20', 'value')])
def update_output(value):
    out = "V20 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV21', 'children'),
    [Input('SV21', 'value')])
def update_output(value):
    out = "V21 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV22', 'children'),
    [Input('SV22', 'value')])
def update_output(value):
    out = "V22 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV23', 'children'),
    [Input('SV23', 'value')])
def update_output(value):
    out = "V23 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV24', 'children'),
    [Input('SV24', 'value')])
def update_output(value):
    out = "V24 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV25', 'children'),
    [Input('SV25', 'value')])
def update_output(value):
    out = "V25 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV26', 'children'),
    [Input('SV26', 'value')])
def update_output(value):
    out = "V26 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV27', 'children'),
    [Input('SV27', 'value')])
def update_output(value):
    out = "V27 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TV28', 'children'),
    [Input('SV28', 'value')])
def update_output(value):
    out = "V28 is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('TAmount', 'children'),
    [Input('SAmount', 'value')])
def update_output(value):
    out = "Amount is set to " + str(value)
    return html.P(out)
@app.callback(
    Output('THour', 'children'),
    [Input('SHour', 'value')])
def update_output(value):
    out = "Hour is set to " + str(value)
    return html.P(out)
# live updater ends here
if __name__ == '__main__':
    app.run_server(debug=True)