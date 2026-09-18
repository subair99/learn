import requests
import functions


url = 'http://localhost:9696/predict'


df = {
    'record': 101.0,
    '0_pre-RR': 315.0,
    '0_post-RR': 321.0,
    '0_pPeak': -0.0621513440809614,
    '0_tPeak': -0.2969825451600668,
    '0_rPeak': 0.9918590560568558,
    '0_sPeak': -0.410306180553059,
    '0_qPeak': -0.0656863278763177,
    '0_qrs_interval': 22.0,
    '0_pq_interval': 3.0,
    '0_qt_interval': 32.0,
    '0_st_interval': 7.0,
    '0_qrs_morph0': -0.0656863278763177,
    '0_qrs_morph1': 0.0514592399077539,
    '0_qrs_morph2': 0.6304189612881775,
    '0_qrs_morph3': 0.8907935853292844,
    '0_qrs_morph4': 0.0912584437242184,
    '1_pre-RR': 315.0,
    '1_post-RR': 321.0,
    '1_pPeak': 0.0213113594631295,
    '1_tPeak': 0.0082455161653336,
    '1_rPeak': 0.0095284734609748,
    '1_sPeak': 0.0082303305375344,
    '1_qPeak': 0.0095284734609748,
    '1_qrs_interval': 3.0,
    '1_pq_interval': 8.0,
    '1_qt_interval': 12.0,
    '1_st_interval': 1.0,
    '1_qrs_morph0': 0.0095284734609748,
    '1_qrs_morph1': 0.0095284734609748,
    '1_qrs_morph2': 0.0087855532848716,
    '1_qrs_morph3': 0.0087855532848716,
    '1_qrs_morph4': 0.0083683934328459
}

patient = functions.process_full(df)

response = requests.post(url, json=patient).json()
print(response)

if response['arrhythmia'] == True:
    print('THE PATIENT HAS ARRHYTHMIA')
else:
    print('THE PATIENT HAS NO ARRHYTHMIA')
