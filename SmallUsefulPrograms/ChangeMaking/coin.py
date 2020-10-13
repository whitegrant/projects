#Grant White
#September 2019


"""
Calculates the least number of coins required to reach a certain value.
    takes two parameters:
        v = the value
        coins = the set of possible coins, defaults to USD coin system
"""
usd = (1, 5, 10, 25, 50, 100)
lookup = {0:0}

def least(value, coins=usd):
    if value in lookup.keys():
        return lookup[value]
    else:
        answer = 1 + min([least(value-coin) for coin in coins if coin <= value])
        lookup[value] = answer
        return answer
