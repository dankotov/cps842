def user_boolean_selection(query, confirmation, rejection):
    response = input(query)
    if response.lower() == "y":
        print(confirmation)
        return True
    elif response.lower() == "n":
        print(rejection)
        return False
    return user_boolean_selection(query, confirmation, rejection)