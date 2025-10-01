import os
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from bybit_client import BybitClient

load_dotenv()

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('webhook.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

bybit_client = BybitClient(
    api_key=os.getenv('BYBIT_API_KEY'),
    api_secret=os.getenv('BYBIT_API_SECRET'),
    testnet=os.getenv('BYBIT_TESTNET', 'false').lower() == 'true'
)

@app.route('/')
def home():
    return jsonify({
        'status': 'running',
        'service': 'TradingView to Bybit Webhook Bridge',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.get_json()
        
        if not data:
            logger.error('No JSON data received')
            return jsonify({'status': 'error', 'message': 'No JSON data'}), 400
        
        logger.info(f'Received webhook: {json.dumps(data, indent=2)}')
        
        event = data.get('event')
        
        if event == 'open_block':
            result = handle_open_block(data)
            return jsonify({'status': 'success', 'result': result}), 200
            
        elif event == 'cancel_block':
            result = handle_cancel_block(data)
            return jsonify({'status': 'success', 'result': result}), 200
            
        else:
            logger.warning(f'Unknown event type: {event}')
            return jsonify({'status': 'error', 'message': f'Unknown event: {event}'}), 400
            
    except Exception as e:
        logger.error(f'Error processing webhook: {str(e)}', exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

def handle_open_block(data):
    symbol = data.get('symbol')
    side = data.get('side')
    legs = data.get('legs', [])
    oid_prefix = data.get('oid_prefix')
    
    logger.info(f'Opening block: {side} {symbol} with {len(legs)} legs, prefix: {oid_prefix}')
    
    results = []
    
    for leg in legs:
        leg_id = leg.get('id')
        order_link_id = leg.get('orderLinkId')
        price = float(leg.get('price'))
        qty = float(leg.get('qty'))
        leverage = leg.get('lev')
        tp = float(leg.get('tp')) if leg.get('tp') else None
        sl = float(leg.get('sl')) if leg.get('sl') else None
        
        logger.info(f'Processing leg {leg_id}: price={price}, qty={qty}, lev={leverage}, tp={tp}, sl={sl}')
        
        try:
            result = bybit_client.place_order(
                symbol=symbol,
                side=side,
                order_type='Limit',
                qty=str(qty),
                price=str(price),
                order_link_id=order_link_id,
                leverage=leverage,
                take_profit=str(tp) if tp else None,
                stop_loss=str(sl) if sl else None
            )
            
            results.append({
                'leg_id': leg_id,
                'order_link_id': order_link_id,
                'status': 'success',
                'result': result
            })
            
            logger.info(f'Leg {leg_id} placed successfully: {result}')
            
        except Exception as e:
            logger.error(f'Error placing order for leg {leg_id}: {str(e)}')
            results.append({
                'leg_id': leg_id,
                'order_link_id': order_link_id,
                'status': 'error',
                'error': str(e)
            })
    
    return results

def handle_cancel_block(data):
    symbol = data.get('symbol')
    oid_prefix = data.get('oid_prefix')
    
    logger.info(f'Cancelling block for symbol {symbol} with prefix: {oid_prefix}')
    
    try:
        result = bybit_client.cancel_orders_by_prefix(symbol, oid_prefix)
        logger.info(f'Cancel result: {result}')
        return result
    except Exception as e:
        logger.error(f'Error cancelling orders: {str(e)}')
        raise

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
