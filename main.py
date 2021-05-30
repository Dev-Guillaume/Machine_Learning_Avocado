warnings.filterwarnings('ignore')

df = pd.read_csv('./data/avocado.csv', index_col = 0)

fig = plt.figure()

st.image("./assets/avocado.png", width = 100)

st.title("Prediction Avocado Price")

st.write("1) Select a region (All for the world)")
region = st.selectbox("Region", ("All", "NewYork", "SanFrancisco", "GrandRapids" , "SanDiego", "Columbus", "Roanoke", "Syracuse", "Philadelphia", "TotalUS", "LasVegas",
"Midsouth", "Sacramento", "Nashville", "SouthCarolina", "GreatLakes", "Indianapolis", "NorthernNewEngland", "Spokane", "Albany", "HarrisburgScranton",
"Detroit", "Orlando", "Jacksonville", "Charlotte", "West", "Denver", "BuffaloRochester", "Plains", "Chicago", "California", "Atlanta", "StLouis",
"RaleighGreensboro", "SouthCentral", "HartfordSpringfield", "PhoenixTucson", "Tampa", "Pittsburgh", "BaltimoreWashington", "Houston", "RichmondNorfolk",   
"Louisville", "CincinnatiDayton", "Northeast", "NewOrleansMobile", "MiamiFtLauderdale", "Southeast", "DallasFtWorth", "Portland", "Boise", "LosAngeles"             
"Boston", "Seattle", "WestTexNewMexico"))

st.write("2) Select a year (All for every year)")
year = st.selectbox("Year", ("All", 2015, 2016, 2017, 2018))

st.write("3) Select a type (All for all types)")
type_avocado = st.selectbox("Type of avocado", ("All", "organic", "conventional"))

def drow_plot(y_test, y_pred):
    plt.plot(y_test, y_pred, color = 'r')
    plt.plot(y_test, y_test, color = 'b')
    plt.title('Result of predicted price of avocado in the world')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.legend(['Actual Price', 'Predicted Price'])
    st.pyplot(fig)

def get_prediction(X_train, X_test, y_train):
    rfg = RandomForestRegressor(max_features = 8, n_estimators = 30)
    rfg.fit(X_train,y_train)
    y_pred = rfg.predict(X_test)
    return y_pred

def set_min_max_scaler(X_train, X_test, y_train, y_test):
    mmscaler = MinMaxScaler()
    X_train = mmscaler.fit_transform(X_train)
    X_test = mmscaler.transform(X_test)
    y_train = LabelEncoder().fit_transform(np.asarray(y_train).ravel())
    y_test = LabelEncoder().fit_transform(np.asarray(y_test).ravel())
    return X_train, X_test, y_train, y_test

def set_variable_x_y(new_df):
    features = ['Total Volume','4046','4225','4770','year','region', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags']
    target = ['AveragePrice']
    X = new_df[features]
    X['region'] = LabelEncoder().fit_transform(X['region'])
    y = new_df[target]
    return X, y

def data_preprocessing(new_df):
    new_df.pop('Date')
    new_df.pop('type')
    return new_df

def set_parameter(new_df, type_filter, parameter):
    if (parameter != 'All'):
        return new_df[(new_df[type_filter] == parameter)]
    else:
        return new_df

def run_ai():
    new_df = df
    new_df = set_parameter(new_df, 'region', region)
    new_df = set_parameter(new_df, 'year', year)
    new_df = set_parameter(new_df, 'type', type_avocado)
    new_df = data_preprocessing(new_df)
    X, y = set_variable_x_y(new_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.80)
    X_train, X_test, y_train, y_test = set_min_max_scaler(X_train, X_test, y_train, y_test)
    y_pred = get_prediction(X_train, X_test, y_train)
    drow_plot(y_test, y_pred)
        
button = st.button('Generate Prediction Sales')
if button:
    run_ai()