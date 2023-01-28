'''
 # @ Create Time: 2023-01-27 18:11:43.035394
'''

import pathlib


from jupyter_dash import JupyterDash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
import numpy as np
import json
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
canadaData = pd.read_csv(DATA_PATH.joinpath("13100096.csv"))


# In[3]:


#canadaData = pd.read_csv("13100096.csv")
provinceCapital = pd.read_csv(DATA_PATH.joinpath("provinceCapital.csv"))
notes = pd.read_csv(DATA_PATH.joinpath('notes.csv'))
notesIndicator = pd.read_csv(DATA_PATH.joinpath('notesIndicators.csv'))
with open(DATA_PATH.joinpath('canada_provinces.geojson')) as data:
   dataset = json.load(data)
provincialData = canadaData[ canadaData['GEO'] != 'Canada (excluding territories)' ]


# In[4]:


ageDict = { idxAge : canadaData['Age group'].unique()[idxAge]  for idxAge in range( canadaData['Age group'].nunique() ) }
sexDict = {idxSex:   canadaData['Sex'].unique()[idxSex]  for idxSex in range( canadaData['Sex'].nunique() ) }
unitsDict = {idxUnit:  canadaData['UOM'].unique()[idxUnit]  for idxUnit in range( canadaData['UOM'].nunique() ) }
yearDict = {0: 2015, 1: 2016, 2: 2017, 3: 2018, 4: 2019, 5: 2020}
#{idxYear:  canadaData['REF_DATE'].unique()[idxYear]  for idxYear in range( canadaData['REF_DATE'].nunique() ) }
variable_clustering = ['None', 'GDP 2020', 'Population', 'GDP per capita']


# In[5]:


yearDict


# In[6]:


merged_df = pd.merge(provincialData,provinceCapital,left_on="GEO",right_on="Province")


# In[7]:


merged_df.head(5)


# In[8]:


app = JupyterDash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP]) 
server = app.server


app.layout = html.Div(
    [
        html.H1('Canada Health dataset', style={'textAlign': 'center'}),
        html.Br(),
        html.H2("Indicator:"),
        dcc.Dropdown(id = 'id_indicator', value="Body mass index, adjusted self-reported, adult (18 years and over), obese",
        options= [ {'label':ind, 'value':ind } for ind in canadaData["Indicators"].unique()]),
        dbc.Row([
                dbc.Col([dbc.Label("Age group:"),
                        dcc.Slider(id = 'id_age',
                        min = 0,
                        max = 5,
                        step=None,
                        dots=True,
                        value = 0,
                        marks= ageDict,
                        included = False)   
                        ], md=2, lg=6),
                dbc.Col([dbc.Label("Sex group:"),
                        dcc.Slider(id='id_sex',
                        min=0,
                        max=2,
                        step=None,
                        dots=True,
                        value = 0,
                        marks = sexDict,
                            included=False)
                        ], md = 10, lg = 3),
                dbc.Col([dbc.Label("Units:"),
                        dcc.Slider(id = "id_units",
                        min=0,
                        max=1,
                        step=None,
                        dots=True,
                        value = 0,
                        marks = unitsDict,
                            included=False)
                        ], md = 10, lg = 3)   
        ]),
        dbc.Row([
                dbc.Col([
                        dcc.Graph("choropleth")
                ], md = 2, lg = 5),
                dbc.Col([
                        dcc.Graph("scatterplot")
                ], md=2, lg=5)
        ]),
        dcc.Markdown(id='indicator_map_details_md', style={'backgroundColor': '#E5ECF6'}),
        dbc.Row([
                dbc.Col([dbc.Label("Select the year:"),
                        dcc.Slider(id = 'id_year',
                        min = 0,
                        max = 5,
                        step=None,
                        dots=True,
                        value = 0,
                        marks= yearDict,
                        included = False)   
                        ], md=2, lg=6),
                dbc.Col([dbc.Label("Select the number of clusters:"),
                        dcc.Slider(id='id_clusters',
                        min=2,
                        max=5,
                        step=None,
                        dots=True,
                        value = 2,
                        marks= { int(bins) : { 'label': str( int(bins) ) } 
                                for bins in np.arange(2,6)},
                            included=False)
                        ], md = 10, lg = 3),
                dbc.Col([dbc.Label("Select an additional variable for clustering:"),
                        dcc.Dropdown(id = "id_variable",
                        value = "GDP 2020",
                        options= [ {'label':variable, 'value':variable } for variable in variable_clustering]),
                        ])                  
        ]),
        dcc.Graph("clusterplot")
    ]
)
def getNotes(indicator):
    references = notesIndicator[notesIndicator['Indicator']==indicator]['Reference'].values[0]
    refAtoms = references.split(';')
    noteList = []
    for note in refAtoms:
        noteList.append( notes[ notes['Note ID']== int(note) ]['Note'].values[0] )
    return noteList

@app.callback(Output(component_id="choropleth", component_property="figure"),
              Input(component_id="id_indicator", component_property="value"),
              Input(component_id="id_age", component_property="value"),
              Input(component_id="id_sex", component_property="value"),
              Input(component_id="id_units", component_property="value"))

def plot_choropleth(indicator, age,sex,units):
        ageDict1 = ageDict[age]
        sexDict1 = sexDict[sex]
        if units==0:
                charValue = 'Number of persons'
        else:
                charValue = 'Percent'

        df = (  provincialData[ (provincialData['Age group'] ==  ageDict1 ) & 
                (provincialData['Sex'] ==  sexDict1 )
              & (provincialData['Characteristics'] == charValue) 
              & (provincialData['Indicators'] == indicator) ]  )                        
        fig=px.choropleth(df,
             geojson=dataset,
             featureidkey='properties.name', #used for the join between the geojson file and the the dataframe
             locations='GEO',  #used for the join between the geojson file and the the dataframe
             color='VALUE',
             color_continuous_scale='Inferno',
             title=indicator ,  
             height=600,
             scope='north america',
             animation_frame='REF_DATE'
              )
        return fig
@app.callback(Output(component_id="scatterplot", component_property="figure"),
              Output(component_id='indicator_map_details_md', component_property="children"),  
              Input(component_id="id_indicator", component_property="value"),
              Input(component_id="id_age", component_property="value"),
              Input(component_id="id_sex", component_property="value"),
              Input(component_id="id_units", component_property="value"))

def plot_scatterplot(indicator, age,sex,units):
        ageDict1 = ageDict[age]
        sexDict1 = sexDict[sex]
        if units==0:
                charValue = 'Number of persons'
        else:
                charValue = 'Percent'

        df = (  merged_df[ (merged_df['Age group'] ==  ageDict1 ) & 
                (merged_df['Sex'] ==  sexDict1 )
              & (merged_df['Characteristics'] == charValue) 
              & (merged_df['Indicators'] == indicator) ]  )

        fig2 = px.scatter_mapbox( df,
                    lon= 'Longitude',
                    lat= 'Latitude',
                    zoom=1,
                    size='VALUE', 
                    size_max=80,
                    color='Region',
                    animation_frame='REF_DATE',
                    opacity=0.7,
                    height=650,
                    hover_data=['Region'],
                    color_discrete_sequence=px.colors.qualitative.G10,
                    mapbox_style='open-street-map',
                    hover_name=df['GEO'],
                    title=None )
        series_df = merged_df[merged_df['Indicators'] == indicator]
        ## MARKDOWN
        #fig2.layout.coloraxis.colorbar.title =getNotes(indicator)
        #markdown = getNotes(indicator)
        notes = getNotes(indicator)
        markdown = "# "+indicator+":\n"
    
        for note in notes:
                markdown+="* "+note+"\n"

        return fig2,markdown

@app.callback(Output(component_id="clusterplot", component_property="figure"),
              Input(component_id="id_indicator", component_property="value"),
              Input(component_id="id_age", component_property="value"),
              Input(component_id="id_sex", component_property="value"),
              Input(component_id="id_units", component_property="value"),
              Input(component_id="id_year", component_property="value"),
              Input(component_id="id_clusters", component_property="value"),
              Input(component_id="id_variable", component_property="value"))

def plot_map_cluster(indicator, age,sex,units, year, clusters, cvar):
        
        variable = ["VALUE"]
        if not cvar == "None":
                variable.append(cvar)

        ageDict1 = ageDict[age]
        sexDict1 = sexDict[sex]
        if units==0:
                charValue = 'Number of persons'
        else:
                charValue = 'Percent'
        #df = pd.merge(provincialData,provinceCapital,left_on="GEO",right_on="Province")
        #df = merged_df.loc[merged_df['REF_DATE']==year][ variables + ['GDP 2020', 'Population', 'GDP per capita'] ]
        
        print(indicator)
        df = provincialData.merge(provinceCapital, left_on="GEO", right_on="Province", how="left")
        
        print( "aaaaa")
        print(yearDict[ year] )
        #print(df)
        
        df1 =   df[ (df['Age group'] ==  ageDict1 ) 
                     & (df['Sex'] ==  sexDict1 ) 
                     & (df['Characteristics'] == charValue) 
                     & (df['Indicators'] == indicator)  
                     & (df["REF_DATE"] == yearDict[ year] )] #.sort_values('REF_DATE')
        
        
        # & (merged_df['REF_DATE']==year) ])#[variable + ['GDP 2020', 'Population', 'GDP per capita']]  )

        arrayData = df1[variable]
        if arrayData.isna().all().any():
                return px.choropleth(title='No available data for the selected combination of year/indicators.')
       
        
        
        # imputing
        imputeData = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform( arrayData )
                
                
        #scaling
        scaledData = StandardScaler().fit_transform( imputeData )
        # kmean
        print(scaledData)
        print(clusters)
        kmeans = KMeans(n_clusters=clusters)
        kmeans.fit(scaledData)

        print( "bbbbb")
        print(kmeans.labels_)
        fig3 = px.choropleth(df1, 
                        #locations="Country Name", 
                        #locationmode='country names',
                        geojson=dataset,
                        featureidkey='properties.name',
                        locations = "GEO",
                        scope="north america",
                        color=[str(x) for x in kmeans.labels_],
                        hover_data=variable,
                        height=650,
                        title=f'Country clusters - {year}. Number of clusters: {clusters}<br>Inertia: {kmeans.inertia_:,.2f}',
                        color_discrete_sequence=px.colors.qualitative.T10 )

        return fig3           
        

#app.run_server(mode='external', height=600, width='80%', port=1235 )


# In[10]:


#server = app.server

if __name__ == "__main__":
    app.run_server(debug = False)