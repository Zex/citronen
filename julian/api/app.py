# Julian REST server
# Author: Zex Li <top_zlynch@yahoo.com>
from flask_restful import Resource, Api
from flask import Flask
from julian.api.v1 import *

app = Flask(__name__)
api = Api(app)

api.add_resource(TechDomain, '/api/v1/tech_domain')
api.add_resource(Industry, '/api/v1/industry')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='12306')
