# Julian REST server
# Author: Zex Li <top_zlynch@yahoo.com>
from flask_restful import Resource, Api
from flask_cors import CORS
from flask import Flask
#from julian.api.v1.industry import Industry
from julian.api.v1.tech_domain import TechDomain

app = Flask(__name__)
CORS(app, supports_credentials=True)
api = Api(app)

api.add_resource(TechDomain, '/api/v1/tech_domain')
#api.add_resource(Industry, '/api/v1/industry')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='12306')
