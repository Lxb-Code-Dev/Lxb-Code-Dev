from sys import exit
from bitcoin.core.script import *
from bitcoin.wallet import CBitcoinSecret, P2PKHBitcoinAddress

from utils import *
from config import my_private_key, my_public_key, my_address, faucet_address
from ex1 import send_from_P2PKH_transaction


cust1_private_key = CBitcoinSecret(
    'cUmiWc8aWHpVtpEWq6evfAUjjNCoF7hdu4RJ8GdkU7hAgw4a17Qg')
#Address: n1JBxkj6RxcSeBiUPDDi6gNJB4U6ym1rte
cust1_public_key = cust1_private_key.pub
cust2_private_key = CBitcoinSecret(
    'cS767S3jcD4oZ8bYYMNfW597sLSQnkw4ZVAiHMfmkfk2mFFwvxw3')
#Address: mpFbpXEV8NEbJVk724m3YYqkNHZAn7Zz4x
cust2_public_key = cust2_private_key.pub
cust3_private_key = CBitcoinSecret(
    'cRiKRhYCwBqCH6krcd1X6tpoDHuR5U59xZYQjAM23uhj2qwy4y46')
#Address: mg9PmTu9VUeWy8aXmQCwY44pzTPfx1qJnT
cust3_public_key = cust3_private_key.pub

######################################################################
# TODO: Complete the scriptPubKey implementation for Exercise 2

# You can assume the role of the bank for the purposes of this problem
# and use my_public_key and my_private_key in lieu of bank_public_key and
# bank_private_key.

ex2a_txout_scriptPubKey = [2,my_public_key,cust1_public_key,cust2_public_key,cust3_public_key,4,OP_CHECKMULTISIG]
######################################################################

if __name__ == '__main__':
    ######################################################################
    # TODO: set these parameters correctly
    amount_to_send = 0.00008
    txid_to_spend = (
        '394177e9c2e971c99b20781c2d970a5ff6a482ef871216746b9db4e350967558')
    utxo_index = 1
    ######################################################################

    response = send_from_P2PKH_transaction(
        amount_to_send, txid_to_spend, utxo_index,
        ex2a_txout_scriptPubKey)
    print(response.status_code, response.reason)
    print(response.text)
