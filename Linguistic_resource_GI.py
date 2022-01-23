import csv
import numpy as np
import re
#http://www.wjh.harvard.edu/~inquirer/homecat.htm
#http://www.wjh.harvard.edu/~inquirer/
#http://www.wjh.harvard.edu/~inquirer/spreadsheet_guide.htm

class GI(object):
    gidict={}
    gidictstartwith={}
        
    def __init__(self):
        csvfile = open('resources/inquirerbasic.txt', 'r')
        lines = csv.reader(csvfile,delimiter="\t")
        self.pattern_split = re.compile(r"\W+")
        for l in lines:
            key= l[0]
            value=l[2],l[3],l[7],l[8],l[14],l[15]
            #positiv,negativ,hostile,strong,pleasur,pain
            if "#" in key:
                self.gidictstartwith.setdefault(key,[])
                self.gidictstartwith[key].append(value)
            else:
                self.gidict.setdefault(key, [])
                self.gidict[key].append(value)
            
        return


    def get_gi(self,text):
        tokens= self.pattern_split.split(text.upper())
        pos=[0];neg=[0];host=[0];strong=[0];pleasure=[0];pain=[0]
        #print tokens
        for word in tokens:
            #print word.upper()
            if word.upper() in self.gidict:

                if self.gidict[word.upper()][0][0]=="":pos.append(0)
                else:pos.append(1)
                if self.gidict[word.upper()][0][1]=="":neg.append(0)
                else:neg.append(1)
                if self.gidict[word.upper()][0][2]=="":host.append(0)
                else:host.append(1)
                if self.gidict[word.upper()][0][3]=="":strong.append(0)
                else:strong.append(1)
                if self.gidict[word.upper()][0][4]=="":pleasure.append(0)
                else:pleasure.append(1)
                if self.gidict[word.upper()][0][5]=="":pain.append(0)
                else:pain.append(1)
            else:
                i=0
                for key,val in self.gidictstartwith.items():
                    if word.upper().startswith(key[:-2]) and i==0:

                        if val[0][0]=="":pos.append(0)
                        else:pos.append(1)
                        if val[0][1]=="":neg.append(0)
                        else:neg.append(1)
                        if val[0][2]=="":host.append(0)
                        else:host.append(1)
                        if val[0][3]=="":strong.append(0)
                        else:strong.append(1)
                        if val[0][4]=="":pleasure.append(0)
                        else:pleasure.append(1)
                        if val[0][5]=="":pain.append(0)
                        else:pain.append(1)
                        i=1
                    elif word.upper().startswith(key[:-2]) and i==1:

                        if val[0][0]!="" and pos[-1]!=1 : pos.append(1)
                        if val[0][1]!="" and neg[-1]!=1 :neg.append(1)
                        if val[0][2]!="" and host[-1]!=1 :host.append(1)
                        if val[0][3]!="" and strong[-1]!=1 :strong.append(1)
                        if val[0][4]!="" and pleasure[-1]!=1 :pleasure.append(1)
                        if val[0][5]!="" and pain[-1]!=1 :pain.append(1)
                        
                        

        return np.sum(pos),np.sum(neg),np.sum(host),np.sum(strong),np.sum(pleasure),np.sum(pain)

        
                
if __name__=='__main__':
    gi=GI()








        
