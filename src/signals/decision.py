def _decide_action(buy_res, sell_res, buy_thr, sell_thr):
    m_buy  = buy_res["score"]  - float(buy_thr)
    m_sell = sell_res["score"] - float(sell_thr)
    if buy_res["buy"] and not sell_res["sell"]:
        return "BUY", m_buy
    if sell_res["sell"] and not buy_res["buy"]:
        return "SELL", m_sell
    if buy_res["buy"] and sell_res["sell"]:
        return ("BUY", m_buy) if m_buy >= m_sell else ("SELL", m_sell)
    return "HOLD", max(m_buy, m_sell)