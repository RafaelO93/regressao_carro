abc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def region(vin):
    if vin[:1] in abc:
        if vin[:1] in abc[:8]:
            return "Africa"
        elif vin[:1] in abc[9:18]:
            return "Asia"
        elif vin[:1] in abc[18:]:
            return "Europa"
    else:
        if int(vin[:1]) in range(1,6):
            return "America do Norte"
        elif int(vin[:1]) in [6,7]:
            return "Oceania"
        elif int(vin[:1]) in [8,9]:
            return "America do Sul"


    