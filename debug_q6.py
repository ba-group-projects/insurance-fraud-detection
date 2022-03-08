import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

pd.set_option('display.max_columns', None)
# read data and get a brief idea of the data
df = pd.read_csv('./materials/Insurance_claims.csv')
# get useful features that needed in the machine learning model
# TODO using nlp to insurer notes data
needed_columns = ['PolicyholderOccupation',
                  'LossDate', 'FirstPolicySubscriptionDate', 'ClaimCause',
                  'ClaimInvolvedCovers', 'DamageImportance', 'FirstPartyVehicleType',
                  'ConnectionBetweenParties', 'PolicyWasSubscribedOnInternet',
                  'NumberOfPoliciesOfPolicyholder', 'FpVehicleAgeMonths',
                  'EasinessToStage', 'ClaimWihoutIdentifiedThirdParty', 'ClaimAmount',
                  'LossHour', 'PolicyHolderAge', 'NumberOfBodilyInjuries',
                  'FirstPartyLiability', 'LossAndHolderPostCodeSame', 'Fraud']
df = df[needed_columns]

# show the first 5 rows, get some idea of the data structure
print(f'Data sample:')
print(df.head(5))  # TODO use sentiment analysis
print('-----------------------------------------------------')

# get the columns name
print('Data Columns:')
print(str(df.columns))
print('-----------------------------------------------------')

# get some basic information about the data, and we found that the min number
# of FpVehicleAgeMonths is less than 0, which does't make sense. We are going
# to detect whether these rows are fraud cases or not. If they are all non-fraud,
# we can drop the rows with negative FpVehicleAgeMonths value. Otherwise, we will
# create a new feature that record these abnormal rows since these unreasonable
# values might be evidence of fraud cases.
print('Data description:')
print(df.describe())
print('-----------------------------------------------------')

# check whether there are any duplicated rows and we found there are 8 duplicated rows,
# which we are going to drop.
print('Data duplicated rows:')
print(df.duplicated().sum())
print('-----------------------------------------------------')


print('The shape of the data before dropping duplicated rows:')
df_shape_before_drop = df.shape
print(df.shape)
print('-----------------------------------------------------')

# drop the duplicated rows
df.drop_duplicates(inplace=True)

print('The shape of the data after dropping duplicated rows:')
df_shape_after_drop = df.shape
print(df.shape)
print('-----------------------------------------------------')

print(
    f'The number of rows that are dropped: {df_shape_before_drop[0]-df_shape_after_drop[0]}')

# check whether unreasonable rows contain fraud cases
df_unreasonable_rows = df[df['FpVehicleAgeMonths'] < 0]
df_shape_before_drop = df_unreasonable_rows.shape
print(df_unreasonable_rows)
print('-----------------------------------------------------')
# we can find that these three rows are not fraud cases.
# Since we have enough non-fraud data, we can drop these rows.
df_shape_before_drop = df.shape
df.drop(df_unreasonable_rows.index, inplace=True)
df_shape_after_drop = df.shape

print(
    f'The number of rows that are dropped: {df_shape_before_drop[0]-df_shape_after_drop[0]}')

# Check how much NaN values in each column.
print(f'Number of NaN values in each column:')
print(df.isnull().sum())

# Check the number of missing data when Frand is True
df_fraud = df[df["Fraud"] == 1]
print(f'Number of NaN values in each column when Frand is True:')
print(df_fraud.isnull().sum())

df_non_frand = df[df["Fraud"] == 0]
print(f'Number of NaN values in each column when Frand is False:')
print(df_non_frand.isnull().sum())

dummy_columns = ['PolicyholderOccupation', 'ClaimCause', 'ClaimInvolvedCovers', 'DamageImportance',
                 'FirstPartyVehicleType', 'ConnectionBetweenParties', 'PolicyWasSubscribedOnInternet']
df[dummy_columns] = df[dummy_columns].fillna('NaN')

df_shape_before_drop = df.shape
df_fraud_shape_before_drop = df[df["Fraud"] == 1].shape
df.dropna(subset=["LossHour", "PolicyHolderAge",
          "FpVehicleAgeMonths"], inplace=True)
df_shape_after_drop = df.shape
df_fraud_shape_after_drop = df[df["Fraud"] == 1].shape
print(
    f'The number of rows that are dropped: {df_shape_before_drop[0]-df_shape_after_drop[0]}')
print(
    f'The number of rows that are dropped when Frand is True: {df_fraud_shape_before_drop[0]-df_fraud_shape_after_drop[0]}')


print('The number of NaN values in each column:')
df.isna().sum()
print('-----------------------------------------------------')

print('The shape of final datasets:')
print(df.shape)
print('-----------------------------------------------------')

print("Data sample:")
print(df.head(5))
print('-----------------------------------------------------')

# turn the date into timestamp
df['LossDate'] = df['LossDate'].apply(
    lambda x: datetime.datetime.strptime(x, '%d.%M.%y').timestamp())
df['FirstPolicySubscriptionDate'] = df['FirstPolicySubscriptionDate'].apply(
    lambda x: datetime.datetime.strptime(x, '%d.%M.%y').timestamp())

# Get dummy variables for categorical data
dummy_columns = ['PolicyholderOccupation', 'ClaimCause', 'DamageImportance',
                 'FirstPartyVehicleType', 'ConnectionBetweenParties', 'PolicyWasSubscribedOnInternet']
# Dummy variables for categorical data
df = pd.get_dummies(df, columns=dummy_columns, drop_first=True)
df

# Extract ClaimInvolvedCovers data
# Get all covers
all_unique = df["ClaimInvolvedCovers"].unique().tolist()
all_covers = str.join(' ', all_unique)  # join the string to list
all_covers_set = set(all_covers.split())  # use set to drop duplicate covers
print(all_covers_set)

for cover in all_covers_set:
    df[f"ClaimInvolvedCovers_{cover}"] = df["ClaimInvolvedCovers"].apply(
        lambda x: 1 if cover in x else 0)
df = df.drop(columns=['ClaimInvolvedCovers'])


# ------------------------------------------------------------
df_autoencoder = df.copy()

# split dataset into non-fraud (normal) and fraud
normal_df = df_autoencoder[df_autoencoder.Fraud == 0]
fraud_df = df_autoencoder[df_autoencoder.Fraud == 1]
print(f'The length of normal_df: {len(normal_df)}')
print(f'The length of fraud_df: {len(fraud_df)}')


# variables for splitting data into train, val and test sets
train_split = 0.8
test_split = 1 - train_split
normal_train_size = round(len(df_autoencoder) * train_split)
fraud_val_test_size = int(len(fraud_df) / 2)
normal_val_test_size = int(
    round(len(df_autoencoder) * test_split / 2) - fraud_val_test_size)
print(f'normal_train_size: {normal_train_size}')
print(f'fraud_val_test_size: {fraud_val_test_size}')
print(f'normal_val_test_size: {normal_val_test_size}\n')

# sample non-fraud data for train set
train_df = normal_df.sample(normal_train_size, random_state=0)
normal_df = normal_df[~normal_df.isin(train_df)].dropna()
print(f'len(train_df): {len(train_df)}')
print(f'len(normal_df) excluding data in train_df: {len(normal_df)}\n')

# sample non-fraud data for val and test sets
val_df = normal_df.sample(normal_val_test_size, random_state=0)
test_df = normal_df[~normal_df.isin(val_df)].dropna()
print(f'len(val_df): {len(val_df)}')
print(f'len(test_df): {len(test_df)}\n')

# check if all normal data is in the train, val and test sets
normal_df = df_autoencoder[df_autoencoder.Fraud == 0]
test = pd.concat([train_df, val_df, test_df]).isin(normal_df)
print(
    f'len(test[test.LossDate == False]) = {len(test[test.LossDate == False])}\n')

# sample fraud data for val and test sets
val_df = val_df.append(fraud_df.sample(fraud_val_test_size, random_state=0))
test_df = test_df.append(fraud_df[~fraud_df.isin(val_df)].dropna())
print(f'len(val_df): {len(val_df)}')
print(f'len(test_df): {len(test_df)}\n')

print(
    f'Check if len(train_df) + len(val_df) + len(test_df) == len(df_autoencoder): {len(train_df) + len(val_df) + len(test_df) == len(df_autoencoder)}')

y_train = train_df[['Fraud']].to_numpy()
y_val = val_df[['Fraud']].to_numpy()
y_test = test_df[['Fraud']].to_numpy()

X_train = train_df.drop(['Fraud'], axis=1)
X_val = val_df.drop(['Fraud'], axis=1)
X_test = test_df.drop(['Fraud'], axis=1)

train_col_names = list(X_train.columns)+['df_key']

# Fit the MinMaxScaler
scaler = MinMaxScaler()

X_train.loc[:, 'df_key'] = 1
X_val.loc[:, 'df_key'] = 0
X_test.loc[:, 'df_key'] = 0
X_train_val = pd.concat([X_train, X_val])
# X_df_key = X_train_val[['df_key']]
# X_train_val = X_train_val.drop(['df_key'], axis=1)

X_train_val = pd.DataFrame(scaler.fit_transform(
    X_train_val), columns=train_col_names)
X_test = pd.DataFrame(scaler.transform(X_test), columns=train_col_names)
X_test = X_test.drop(['df_key'], axis=1)
X_train = X_train_val[X_train_val.df_key == 1].drop(['df_key'], axis=1)
X_val = X_train_val[X_train_val.df_key == 0].drop(['df_key'], axis=1)

print(
    f'check if len(X_train) + len(X_val) + len(X_test) == len(df_autoencoder): {len(X_train) + len(X_val) + len(X_test) == len(df_autoencoder)}')
print(f"X_train.shape: {X_train.shape}")
print(f"X_val.shape: {X_val.shape}")
print(f"X_test.shape: {X_test.shape}")
X_train = X_train.to_numpy()
X_val = X_val.to_numpy()
X_test = X_test.to_numpy()

X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(
    X_train, y_train, test_size=0.3, random_state=0)


class AutoEncodingModel:
    def __init__(self, reg_param, number_units_layers, number_units_bottleneck, dropout_rate, number_layers, whether_dropout, whether_regularizer, X_train_train=X_train_train, X_train_val=X_train_val, X_val=X_val) -> None:
        tf.keras.backend.clear_session()
        tf.random.set_seed(48)
        self.X_train_train = X_train_train
        self.X_train_val = X_train_val
        self.X_val = X_val
        self.reg_param = reg_param
        self.number_units_layers = number_units_layers
        self.number_units_bottleneck = number_units_bottleneck
        self.dropout_rate = dropout_rate
        self.number_layers = number_layers
        self.whether_dropout = whether_dropout
        self.whether_regularizer = whether_regularizer
        self.input_dim = self.X_train_train.shape[1]
        self.build_model()
        self.compile_model()

    def build_model(self):
        # if whether_regularizer == True, then reg_param is used or it will be 0
        regularizer = tf.keras.regularizers.l2(
            self.reg_param*self.whether_regularizer)
        if self.whether_dropout == True:
            encoder = tf.keras.models.Sequential([
                tf.keras.layers.Dense(self.number_units_layers, activation="relu")]*self.number_layers+[
                tf.keras.layers.Dropout(self.dropout_rate), # dropout before the bottleneck layer
                tf.keras.layers.Dense(self.number_units_bottleneck,activation='sigmoid', kernel_regularizer=regularizer)])
        else:
            encoder = tf.keras.models.Sequential([
                tf.keras.layers.Dense(self.number_units_layers, activation="relu")]*self.number_layers+[
                tf.keras.layers.Dense(self.input_dim,activation='sigmoid')])
        decoder = tf.keras.models.Sequential([
                tf.keras.layers.Dense(self.number_units_layers, activation="relu")]*self.number_layers+[
                tf.keras.layers.Dense(self.input_dim, activation="sigmoid")])
        self.autoencoder = tf.keras.models.Sequential([encoder, decoder])

        # random_seed = 192
    def get_hp(self):
        # get all hyperparameters
        return {
            'reg_param': self.reg_param,
            'number_units_layers': self.number_units_layers,
            'number_units_bottleneck': self.number_units_bottleneck,
            'dropout_rate': self.dropout_rate,
            'number_layers': self.number_layers,
            'whether_dropout': self.whether_dropout,
            'whether_regularizer': self.whether_regularizer
        }

    def get_model(self):
        # get the model
        return self.autoencoder

    def compile_model(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.autoencoder.compile(
            optimizer=optimizer, loss='mean_squared_error')

    def __train_model_non_fraud(self):
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)
        self.log_train_non_fraud = self.autoencoder.fit(x=self.X_train_train, y=self.X_train_train,
                                                        epochs=100,
                                                        validation_data=(X_train_val, X_train_val), callbacks=[early_stopping_cb])

    def get_train_non_fraud_loss_diff(self):
        self.__train_model_non_fraud()
        return self.log_train_non_fraud.history['train_loss'] - self.log_train_non_fraud.history['val_loss']

    def __apply_model_fraud(self):
        reconstructions = autoencoder.predict(X_val)
        self.val_loss = tf.keras.losses.mae(reconstructions, X_val)

    def get_val_loss(self):
        self.__apply_model_fraud()
        return self.val_loss

    def run(self):
        non_frand_loss = self.get_train_non_fraud_loss_diff()
        fraud_loss = self.get_val_loss()
        return {"non_frand_loss": non_frand_loss, "fraud_loss": fraud_loss}


autoencoder = AutoEncodingModel(reg_param=0.01, number_units_layers=60, number_units_bottleneck=30,
                                dropout_rate=0.2, number_layers=2, whether_dropout=True, whether_regularizer=True)
res = autoencoder.run()
print(res)
