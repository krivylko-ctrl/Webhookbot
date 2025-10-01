import logging
from pybit.unified_trading import HTTP

logger = logging.getLogger(__name__)

class BybitClient:
    def __init__(self, api_key, api_secret, testnet=False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        if not api_key or not api_secret:
            raise ValueError('BYBIT_API_KEY and BYBIT_API_SECRET must be set')
        
        self.session = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret
        )
        
        logger.info(f'Bybit client initialized (testnet={testnet})')
    
    def set_leverage(self, symbol, leverage):
        try:
            result = self.session.set_leverage(
                category='linear',
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage)
            )
            logger.info(f'Leverage set to {leverage} for {symbol}')
            return result
        except Exception as e:
            logger.warning(f'Error setting leverage: {str(e)}')
            return None
    
    def place_order(self, symbol, side, order_type, qty, price, order_link_id, 
                    leverage=None, take_profit=None, stop_loss=None):
        
        if leverage:
            self.set_leverage(symbol, leverage)
        
        order_params = {
            'category': 'linear',
            'symbol': symbol,
            'side': side,
            'orderType': order_type,
            'qty': qty,
            'price': price,
            'orderLinkId': order_link_id,
            'timeInForce': 'GTC'
        }
        
        if take_profit:
            order_params['takeProfit'] = take_profit
        
        if stop_loss:
            order_params['stopLoss'] = stop_loss
        
        logger.info(f'Placing order: {order_params}')
        
        try:
            result = self.session.place_order(**order_params)
            
            if result.get('retCode') == 0:
                logger.info(f'Order placed successfully: {result}')
                return result
            else:
                error_msg = result.get('retMsg', 'Unknown error')
                logger.error(f'Order failed: {error_msg}')
                raise Exception(f'Bybit API error: {error_msg}')
                
        except Exception as e:
            logger.error(f'Exception placing order: {str(e)}')
            raise
    
    def get_open_orders(self, symbol):
        try:
            result = self.session.get_open_orders(
                category='linear',
                symbol=symbol
            )
            return result.get('result', {}).get('list', [])
        except Exception as e:
            logger.error(f'Error getting open orders: {str(e)}')
            return []
    
    def cancel_order(self, symbol, order_link_id):
        try:
            result = self.session.cancel_order(
                category='linear',
                symbol=symbol,
                orderLinkId=order_link_id
            )
            logger.info(f'Cancelled order {order_link_id}: {result}')
            return result
        except Exception as e:
            logger.error(f'Error cancelling order {order_link_id}: {str(e)}')
            raise
    
    def cancel_orders_by_prefix(self, symbol, prefix):
        open_orders = self.get_open_orders(symbol)
        
        cancelled_orders = []
        errors = []
        
        for order in open_orders:
            order_link_id = order.get('orderLinkId', '')
            
            if order_link_id.startswith(prefix):
                try:
                    result = self.cancel_order(symbol, order_link_id)
                    cancelled_orders.append({
                        'orderLinkId': order_link_id,
                        'status': 'cancelled',
                        'result': result
                    })
                except Exception as e:
                    errors.append({
                        'orderLinkId': order_link_id,
                        'status': 'error',
                        'error': str(e)
                    })
        
        logger.info(f'Cancelled {len(cancelled_orders)} orders with prefix {prefix}')
        
        return {
            'cancelled': cancelled_orders,
            'errors': errors,
            'total_cancelled': len(cancelled_orders),
            'total_errors': len(errors)
        }
