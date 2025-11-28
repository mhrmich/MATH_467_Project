import math


def Cox_Ross_Rubinstein_Tree(S, K, T, r, sigma, N, Option_type):
    # Underlying price (per share): S;
    # Strike price of the option (per share): K;
    # Time to maturity (years): T;
    # Continuously compounding risk-free interest rate: r;
    # Volatility: sigma;
    # Number of binomial steps: N;

    # The factor by which the price rises (assuming it rises) = u ;
    # The factor by which the price falls (assuming it falls) = d ;
    # The probability of a price rise = pu ;
    # The probability of a price fall = pd ;
    # discount rate = disc ;

    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = math.exp(-sigma * math.sqrt(dt))
    pu = (math.exp(r * dt) - d) / (u - d)
    pd = 1 - pu
    disc = math.exp(-r * dt)

    St = [0] * (N + 1)
    C = [0] * (N + 1)

    St[0] = S * d ** N

    for j in range(1, N + 1):
        St[j] = St[j - 1] * u / d

    for j in range(0, N + 1):
        if Option_type == "P":
            C[j] = max(K - St[j], 0)
        elif Option_type == "C":
            C[j] = max(St[j] - K, 0)

    for i in range(N, 0, -1):
        for j in range(0, i):
            C[j] = disc * (pu * C[j + 1] + pd * C[j])

    return C[0]


def Jarrow_Rudd_Tree(S, K, T, r, sigma, N, Option_type):
    # Underlying price (per share): S;
    # Strike price of the option (per share): K;
    # Time to maturity (years): T;
    # Continuously compounding risk-free interest rate: r;
    # Volatility: sigma;
    # Steps: N;

    # The factor by which the price rises (assuming it rises) = u ;
    # The factor by which the price falls (assuming it falls) = d ;
    # The probability of a price rise = pu ;
    # The probability of a price fall = pd ;
    # discount rate = disc ;

    dt = T / N
    u = math.exp((r - (sigma ** 2 / 2)) * dt + sigma * math.sqrt(dt))
    d = math.exp((r - (sigma ** 2 / 2)) * dt - sigma * math.sqrt(dt))
    pu = 0.5
    pd = 1 - pu
    disc = math.exp(-r * dt)

    St = [0] * (N + 1)
    C = [0] * (N + 1)

    St[0] = S * d ** N

    for j in range(1, N + 1):
        St[j] = St[j - 1] * u / d

    for j in range(0, N + 1):
        if Option_type == "P":
            C[j] = max(K - St[j], 0)
        elif Option_type == "C":
            C[j] = max(St[j] - K, 0)

    for i in range(N, 0, -1):
        for j in range(0, i):
            C[j] = disc * (pu * C[j + 1] + pd * C[j])

    return C[0]
