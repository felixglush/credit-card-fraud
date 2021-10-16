def get_timestamp_features(transactions_df, src_feature):
    """
    Extracts timestamp features from a source feature.

    :param transactions_df: The dataframe containing transaction data
    :param src_feature: The feature from which to create new features
    :return: transaction_df with features derived from timestamp
    """

    transactions_df['TX_DURING_WEEKEND'] = \
        transactions_df[src_feature].apply(lambda datetime: int(datetime.weekday() >= 5))
    transactions_df['TX_DURING_NIGHT'] = \
        transactions_df[src_feature].apply(lambda datetime: int(datetime.hour <= 6))

    return transactions_df


def get_customer_spending_behaviour_features(customer_transactions, windows_size_in_days=[1, 7, 30]):
    customer_transactions = customer_transactions.sort_values('TX_DATETIME')

    customer_transactions.index = customer_transactions.TX_DATETIME  # this lets us use days in the rolling window

    for window in windows_size_in_days:
        # frequency feature
        num_tx_window = customer_transactions.TX_AMOUNT.rolling(str(window) + 'd').count()
        # monetary feature
        avg_val_window = customer_transactions.TX_AMOUNT.rolling(str(window) + 'd').sum() / num_tx_window

        customer_transactions['CUSTOMER_ID_NB_TX_' + str(window) + 'DAY_WINDOW'] = list(num_tx_window)
        customer_transactions['CUSTOMER_ID_AVG_AMOUNT_' + str(window) + 'DAY_WINDOW'] = list(avg_val_window)

    customer_transactions.index = customer_transactions.TRANSACTION_ID
    return customer_transactions


def apply_customer_features_to_all(transactions_df):
    transactions_customer_features_df = transactions_df.groupby('CUSTOMER_ID').apply(lambda customer:
                                                                                     get_customer_spending_behaviour_features(
                                                                                         customer,
                                                                                         windows_size_in_days=[1, 7,
                                                                                                               30]))

    transactions_customer_features_df.sort_values('TX_DATETIME').reset_index(drop=True)

    return transactions_customer_features_df


def get_count_risk_rolling_window(terminal_transactions, delay_period=7, windows_size_in_days=[1, 7, 30]):
    terminal_transactions = terminal_transactions.sort_values('TX_DATETIME')
    terminal_transactions.index = terminal_transactions.TX_DATETIME  # this lets us use days in the rolling window

    # calculate delay
    num_fraud_tx_delay = terminal_transactions.TX_FRAUD.rolling(str(delay_period) + 'd').sum()
    qty_fraud_tx_delay = terminal_transactions.TX_FRAUD.rolling(str(delay_period) + 'd').count()

    for window in windows_size_in_days:
        # calculate delay+window size
        num_fraud_tx_delay_and_window = terminal_transactions.TX_FRAUD.rolling(str(delay_period + window) + 'd').sum()
        qty_fraud_tx_delay_and_window = terminal_transactions.TX_FRAUD.rolling(str(delay_period + window) + 'd').count()

        # remove effect of the last delay period to create the delayed response to detecting fraudulent/risky terminals
        num_fraud_tx_window = num_fraud_tx_delay_and_window - num_fraud_tx_delay
        qty_fraud_tx_window = qty_fraud_tx_delay_and_window - qty_fraud_tx_delay

        # risk quantifier for the terminal in the window (between 0 and 1)
        risk_window = num_fraud_tx_window / qty_fraud_tx_window

        terminal_transactions['TERMINAL_ID_NB_TX_' + str(window) + 'DAY_WINDOW'] = list(num_fraud_tx_window)
        terminal_transactions['TERMINAL_ID_RISK_' + str(window) + 'DAY_WINDOW'] = list(risk_window)

    terminal_transactions.index = terminal_transactions.TRANSACTION_ID

    # Replace NA values with 0 (all undefined risk scores where NB_TX_WINDOW is 0)
    terminal_transactions.fillna(0, inplace=True)

    return terminal_transactions


def apply_terminal_features_to_all(transactions_customer_features_df):
    transformed = transactions_customer_features_df.groupby('TERMINAL_ID').apply(
        lambda terminals: get_count_risk_rolling_window(terminals, delay_period=7, windows_size_in_days=[1, 7, 30]))
    transformed = transformed.sort_values('TX_DATETIME').reset_index(drop=True)
    return transformed
