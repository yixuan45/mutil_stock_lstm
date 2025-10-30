from basic.algo import *


class SLAVE(BASE):
    def __init__(self, client, paras, algoid):
        super().__init__(algoid, client, paras)

    def slave_orders(self, target_prod):
        trade_queue = []
        for p_list in target_prod:
            trade_sets = self.paras['apisets'].copy()
            current_price = HELPER.ticker_price(p_list['symbol'])
            trade_sets['symbol'] = p_list['symbol']
            trade_sets['qty'] = self.paras['amountusd'] / current_price
            trade_sets['side'] = p_list['side']
            trade_sets['kill_time'] = HELPER.last_ticker_time() + self.paras['hold_time'] * 60000
            trade_queue.append(trade_sets.copy())
        self.base_orders(trade_queue)

    def slave_positions(self, action_info):  # {groupid  : 'stop'/'kill'/take_profit}
        trade_queue = []
        for symbol in action_info:
            trade_sets = self.paras['apisets'].copy()
            if 'stop' in action_info.values():
                trade_sets['type'] = 'MARKET'
                trade_sets['iceberg'] = self.paras['stop_iceberg']
                trade_sets['wait_time'] = self.paras['stop_interval']
            else:
                trade_sets['type'] = self.paras['position_type']
            pos = self.infos['trades'][ symbol]['position']
            trade_sets['symbol'] = symbol
            trade_sets['reduceOnly'] = True
            trade_sets['side'] = 'SELL' if pos > 0 else 'BUY'
            trade_sets['qty'] = abs(pos)
            trade_sets['kill_time'] = self.infos['trades'][ symbol]['kill_time']
            trade_queue.append(trade_sets.copy())

        Logger.DEBUG(f'{self.algoid}:trade_queue', trade_queue)
        self.base_orders(trade_queue)
        Logger.INFO(f'{self.algoid}:infos', self.infos)

        self.base_infos()
        result_cur=[]
        for s in action_info:

            result_cur.append([HELPER.time,self.infos['realized'][s]])
            # 清空当前的利润
            self.base_clear_key(s)
        return result_cur

