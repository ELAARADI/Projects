# -*- coding: utf-8 -*-
"""
This is the second part of the project related to the creation of the API.
"""

import requests
import pandas as pd
from IPython.core.display import HTML
from flask import Flask, render_template, request
from flask_restful import Resource, Api, reqparse

df_funds = pd.read_csv("data_WS.csv",index_col=0)

app = Flask("My App", template_folder = "templates")
api = Api(app)


class Funds(Resource):
    """This is the resource that will be used to filter funds with respect to
    their performance and risk characteristics
    Returns dict"""
    def get(self):
        """function to parse requests made on this resource"""
        parser = reqparse.RequestParser()

        #parser.add_argument('close', required=False)
        parser.add_argument('var_min', required=False)
        parser.add_argument('var_max', required=False)
        parser.add_argument('YoY_min', required=False)
        parser.add_argument('YoY_max', required=False)
        parser.add_argument('risk_min', required=False)
        parser.add_argument('risk_max', required=False)

        args = parser.parse_args()

        # Args is a dictionary
        data = df_funds
        if args['var_min'] is not None:
            data = data[data['fund_var'] >= float(args['var_min'])]
        if args['var_max'] is not None:
            data = data[data['fund_var'] <= float(args['var_max'])]
        if args['YoY_min'] is not None:
            data = data[data['fund_YoY'] >= float(args['YoY_min'])]
        if args['YoY_max'] is not None:
            data = data[data['fund_YoY'] <= float(args['YoY_max'])]
        if args['risk_min'] is not None:
            data = data[data['risk_level'] >= float(args['risk_min'])]
        if args['risk_max'] is not None:
            data = data[data['risk_level'] <= float(args['risk_max'])]

        funds_data = [data.iloc[i].to_dict() for i in range(len(data))]

        return funds_data, 200

api.add_resource(Funds, '/funds')


class FundsId(Resource):
    """This is the resource that will be used to filter funds with respect to
    their name / returns dict"""
    def get(self):
        """function to parse requests made on this resource"""
        parser = reqparse.RequestParser()

        parser.add_argument('ref_max', required=False)
        parser.add_argument('ref_min', required=False)
        parser.add_argument('ref_eq', required=False)

        args = parser.parse_args()

        data_id = df_funds

        if args['ref_max'] is not None:
            if str(args['ref_max']).islower():
                data_id = data_id[data_id["fund_name"].str[0] <= \
                                   str(args['ref_max']).upper()]
            else:
                data_id = data_id[data_id["fund_name"].str[0] <= \
                                   str(args['ref_max'])]

        if args['ref_min'] is not None:
            if str(args['ref_min']).islower():
                data_id = data_id[data_id["fund_name"].str[0] >= \
                                   str(args['ref_min']).upper()]
            else:
                data_id = data_id[data_id["fund_name"].str[0] >= \
                                   str(args['ref_min'])]

        if args['ref_eq'] is not None:
            if str(args['ref_eq']).islower():
                data_id = data_id[data_id["fund_name"].str[0] == \
                                   str(args['ref_eq']).upper()]
            else:
                data_id = data_id[data_id["fund_name"].str[0] == \
                                   str(args['ref_eq'])]

        data_id = data_id.sort_values("fund_name")
        funds_id = [data_id.iloc[i].to_dict() for i in range(len(data_id))]

        return funds_id, 200

api.add_resource(FundsId, '/funds_id')

class FundsManager(Resource):
    """This is the resource that will be used to filter funds with respect to
    their asset manager / returns dict"""
    def get(self):
        """function to parse requests made on this resource"""
        parser = reqparse.RequestParser()

        parser.add_argument('name', required=False)

        args = parser.parse_args()

        data_id = df_funds

        if args['name'] is not None:
            data_id = df_funds[df_funds.fund_name.str.contains(str(args['name']), \
                                                               case = False)]

        funds_id = [data_id.iloc[i].to_dict() for i in range(len(data_id))]

        return funds_id, 200

api.add_resource(FundsManager, '/funds_manager')

@app.route("/")
def home():
    """ Landing page of the API / renders html template"""
    return render_template("code_html_confirme.html")

@app.route("/search", methods=["POST"])
def search():
    """Endpoint used when the form on the initial landing page is filled and submitted"""
    donnees = request.form
    all_numeric = True
    for val in donnees.values():
        if val[0] == '-':
            if val[1:].isnumeric() == False:
                all_numeric = False
                break
        elif val.isnumeric() == False:
            all_numeric = False
            break
    if all_numeric == False:
        return   ("<h1>Error: please go back and input numerical values</h1>"
                  "<a href='http://127.0.0.1:5000/#tailored_search'> "
                  "Please click here to go back </ha")
    else:
        str_rmin = str(donnees['rmin'])
        str_rmax = str(donnees['rmax'])
        str_ymax = str(donnees['Ymax'])
        str_ymin = str(donnees['Ymin'])
        response = requests.get(f"http://127.0.0.1:5000/funds?risk_min={str_rmin}&"+ \
                            f"risk_max={str_rmax}&YoY_max={str_ymax}&YoY_min={str_ymin}")
        json_data = response.json()
        df_resp = pd.DataFrame(json_data)
        return render_template("table_format_confirme.html", data = \
                HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

######### all funds
@app.route("/funds_all")
def all_funds():
    """Endpoint to all funds data"""
    response = requests.get("http://127.0.0.1:5000/funds_id?ref_min=A")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

######### all risks
@app.route("/funds/low_risk")
def low_risk_funds():
    """Endpoint to low risk funds data"""
    response = requests.get("http://127.0.0.1:5000/funds?risk_max=2")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

@app.route("/funds/med_risk")
def med_risk_funds():
    """Endpoint to mid risk funds data"""
    response = requests.get("http://127.0.0.1:5000/funds?risk_min=3&risk_max=5")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

@app.route("/funds/high_risk")
def high_risk_funds():
    """Endpoint to high risk funds data"""
    response = requests.get("http://127.0.0.1:5000/funds?risk_min=6")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

######### all perfs
@app.route("/funds/all_low_perf")
def low_perf_funds():
    """Endpoint to low perf funds data"""
    response = requests.get("http://127.0.0.1:5000/funds?YoY_max=0")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

@app.route("/funds/all_mid_perf")
def med_perf_funds():
    """Endpoint to mid perf funds data"""
    response = requests.get("http://127.0.0.1:5000/funds?YoY_min=0.001&YoY_max=5")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

@app.route("/funds/all_high_perf")
def high_perf_funds():
    """Endpoint to high perf funds data"""
    response = requests.get("http://127.0.0.1:5000/funds?YoY_min=5.001")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

############ combinations risk/perf
# low perf
@app.route("/funds/low_risk_low_perf")
def low_risk_low_perf():
    """Endpoint to low risk/perf funds data"""
    response = requests.get("http://127.0.0.1:5000/funds?risk_max=2&YoY_max=0")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

# mid perf
@app.route("/funds/low_risk_mid_perf")
def low_risk_mid_perf():
    """Endpoint to low risk / mid perf funds data"""
    response = requests.get("http://127.0.0.1:5000/funds?risk_max=2&YoY_min=0.001"+ \
                            "&YoY_max=5")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

@app.route("/funds/mid_risk_mid_perf")
def mid_risk_mid_perf():
    """Endpoint to mid risk / mid perf funds data"""
    response = requests.get("http://127.0.0.1:5000/funds?risk_max=5&risk_min=3&"+ \
                            "YoY_min=0.001&YoY_max=5")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

# High perf
@app.route("/funds/low_risk_high_perf")
def low_risk_high_perf():
    """Endpoint to low risk / high perf funds data"""
    response = requests.get("http://127.0.0.1:5000/funds?risk_max=2&YoY_min=5.001")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

@app.route("/funds/mid_risk_high_perf")
def mid_risk_high_perf():
    """Endpoint to mid risk / high perf funds data"""
    response = requests.get("http://127.0.0.1:5000/funds?risk_max=5&risk_min=3&"+ \
                            "YoY_min=5.001")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

@app.route("/funds/high_risk_high_perf")
def high_risk_high_perf():
    """Endpoint to high risk / high perf funds data"""
    response = requests.get("http://127.0.0.1:5000/funds?risk_min=6&YoY_min=5.001")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

############## Alphabetical order


@app.route("/funds/AD")
def funds_a_d():
    """Endpoint to data of funds with Initial from A-D"""
    response = requests.get("http://127.0.0.1:5000/funds_id?ref_max=D")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

@app.route("/funds/EH")
def funds_e_h():
    """Endpoint to data of funds with Initial from E-H"""
    response = requests.get("http://127.0.0.1:5000/funds_id?ref_min=E&ref_max=H")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

@app.route("/funds/IL")
def funds_i_l():
    """Endpoint to data of funds with Initial from I-L"""
    response = requests.get("http://127.0.0.1:5000/funds_id?ref_min=I&ref_max=L")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

@app.route("/funds/MP")
def funds_m_p():
    """Endpoint to data of funds with Initial from M-P"""
    response = requests.get("http://127.0.0.1:5000/funds_id?ref_min=M&ref_max=P")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

@app.route("/funds/QT")
def funds_q_t():
    """Endpoint to data of funds with Initial from Q-T"""
    response = requests.get("http://127.0.0.1:5000/funds_id?ref_min=Q&ref_max=T ")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

@app.route("/funds/UZ")
def funds_u_z():
    """Endpoint to data of funds with Initial from U-Z"""
    response = requests.get("http://127.0.0.1:5000/funds_id?ref_min=U")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

############## funds per manager

@app.route("/funds/amundi")
def funds_amundi():
    """Endpoint to data of Amundi funds"""
    response = requests.get("http://127.0.0.1:5000/funds_manager?name=Amundi")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

@app.route("/funds/blackrock")
def funds_blackrock():
    """Endpoint to data of Blackrock funds"""
    response = requests.get("http://127.0.0.1:5000/funds_manager?name=blackrock")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

@app.route("/funds/bgf")
def funds_bgf():
    """Endpoint to data of BGF funds"""
    response = requests.get("http://127.0.0.1:5000/funds_manager?name=bgf")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

@app.route("/funds/sg")
def funds_sg():
    """Endpoint to data of SG funds"""
    response = requests.get("http://127.0.0.1:5000/funds_manager?name=sg")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

@app.route("/funds/cpr")
def funds_cpr():
    """Endpoint to data of CPR funds"""
    response = requests.get("http://127.0.0.1:5000/funds_manager?name=cpr")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

@app.route("/funds/edr")
def funds_edr():
    """Endpoint to data of EDR funds"""
    response = requests.get("http://127.0.0.1:5000/funds_manager?name=edr")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

@app.route("/funds/etoilegestion")
def funds_etoilegestion():
    """Endpoint to data of ETOILE funds"""
    response = requests.get("http://127.0.0.1:5000/funds_manager?name=etoile")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

@app.route("/funds/firsteagle")
def funds_firsteagle():
    """Endpoint to data of FirstEagle funds"""
    response = requests.get("http://127.0.0.1:5000/funds_manager?name=eagle")
    json_data = response.json()
    df_resp = pd.DataFrame(json_data)
    return render_template("table_format_confirme.html", data = \
        HTML(df_resp.to_html(classes="table table-stripped active w3-margin-top")))

if __name__=='__main__':
    app.run()
