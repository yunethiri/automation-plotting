import pandas as pd
import matplotlib.figure as fig
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import os
from scipy import stats
pd.options.mode.chained_assignment = None  # default='warn'

# set workingpath
path = "/Users/yunethirikhin/School/Y2 Winter - Research/data"
os.chdir(path)

# to automatically save plots to a folder
outpath = "automation_v2"

# user input
n_factors = 11
optimal_run = 100
user_input_species = "PM2.5"
# list of factors based on input
factors_list = [("Factor " + str(i)) for i in range (1, n_factors +1)]

### READ IN DATA ###

# for piechart, factor contribution and linear regression plots
est_conc = pd.read_excel("Plst18All35RemoveFAFac11_base.xlsx", sheet_name = "Contributions", skiprows = range(3))
# for piechart and linear regression plots 
observed_conc = pd.read_excel("PMF input for Yune.xlsx")
# for factor profile 
profile =  pd.read_excel("Plst18All35RemoveFAFac11_base.xlsx", sheet_name = "Profiles")

# data processing 
est_conc = est_conc.drop(columns = ['Unnamed: 0'])
est_conc.rename(columns = {'Unnamed: 1' : 'Date'}, inplace = True)
#est_conc = est_conc[est_conc["Factor 1"] != -999.0]

# general plot parameters
plt.rcParams["figure.figsize"] = [8.00, 4.00]
plt.rcParams["figure.autolayout"] = True

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 15

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

### PIECHART ###
contributions = est_conc[244:] 
class_data = observed_conc.loc[:,["Date","Class"]]
contributions_piechart = pd.merge(contributions, class_data, on = "Date", how = "left")

# to count NSD & SD days 
counts = class_data["Class"].value_counts()
NSD_counts = counts[0]
SD_counts = counts[1]
overall_counts = NSD_counts + SD_counts

# separate data into classes 
contributions_piechart[factors_list] = contributions_piechart[factors_list].astype(float)
all_data_piechart = contributions_piechart
contributions_NSD = contributions_piechart[contributions_piechart["Class"] == "NSD"]
contributions_SD = contributions_piechart[contributions_piechart["Class"] == "SD"]

# find mean
all_data_mean = all_data_piechart.mean(numeric_only= True)
NSD_mean = contributions_NSD.mean(numeric_only= True)
SD_mean = contributions_SD.mean(numeric_only= True)

# factor names and data for plotting
factors = list(contributions_piechart.columns[1:12])
data_NSD = NSD_mean.tolist()
data_SD = SD_mean.tolist()
all_data_piechart = all_data_mean.tolist()

# autocpt arguments
def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%".format(pct, absolute)

# plot pies
fig, ax1 = plt.subplots()
ax1.pie(data_NSD, labels = factors, autopct = lambda pct: func(pct, data_NSD))
ax1.set_title("NSD (n = " + str(NSD_counts) +")")
plt.draw()
figure = plt.gcf()
figure.set_size_inches(7,5)
#fig.savefig(path.join(outpath, "piechart_NSD.png"), bbox_inches = "tight", dpi = 100)

fig, ax2 = plt.subplots()
ax2.pie(data_SD, labels = factors, autopct = lambda pct: func(pct, data_SD))
ax2.set_title("SD (n = " + str(SD_counts) +")")
plt.draw()
figure = plt.gcf()
figure.set_size_inches(7,5)
#fig.savefig(path.join(outpath, "piechart_SD.png"), bbox_inches = "tight", dpi = 100)

fig, ax3 = plt.subplots()
ax3.pie(all_data_piechart, labels = factors, autopct = lambda pct: func(pct, all_data_piechart))
ax3.set_title("Overall (n = " + str(overall_counts) +")")
plt.draw()
figure = plt.gcf()
figure.set_size_inches(7,5)
#fig.savefig(path.join(outpath, "piechart_overall.png"), bbox_inches = "tight", dpi = 100)

### FACTOR PROFILE ###
conc_data = profile[1:37][2:]
conc_data = conc_data.drop(columns = ['Unnamed: 0'])

# make a list of unnamed columns based on user input
unnamed_list = [("Unnamed: " + str(i)) for i in range (2, n_factors +2)]

# make columns_dict 
columns_dict = {'Unnamed: 1': 'Species'}
temp_dict = dict(zip(unnamed_list, factors_list))
for key, value in temp_dict.items():
    columns_dict[key] = value

conc_data.rename(columns = columns_dict , inplace = True)

pct_data = profile[39:75][2:]
pct_data = pct_data.drop(columns= ['Unnamed: 0'])
pct_data.rename(columns = columns_dict, inplace = True)

# loop through factors_list to generate factor profiles for all factors
species = np.array(pct_data['Species'])

for factor in factors_list:
    factor_conc = np.array(conc_data[factor])
    factor_pct = np.array(pct_data[factor])

    fig, ax4 = plt.subplots()
    ax5 = ax4.twinx()

    ax4.bar(species, factor_conc, color = "blue", width = 0.8, alpha = 0.5, edgecolor = "black", label = "Conc. of Species")
    ax5.plot(species, factor_pct, 'o', color = "red", label = "% of Species")

    ax4.set_xlabel("Species")
    ax4.set_ylabel(u"Conc. of Species (\u03bcg/$m^3$)")
    ax5.set_ylabel("% of Species")

    factor_V = conc_data[conc_data["Species"] == "V"][factor]
    factor_Ni = conc_data[conc_data["Species"] == "Ni"][factor]
    factor_OC = conc_data[conc_data["Species"] == "OC"][factor]
    factor_EC = conc_data[conc_data["Species"] == "EC"][factor]
    V_Ni_Ratio = (factor_V.item()/factor_Ni.item()) if (factor_V.item() != 0 and factor_Ni.item() != 0) else (0)
    OC_EC_Ratio = (factor_OC.item()/factor_EC.item()) if (factor_OC.item() != 0 and factor_EC.item() != 0) else (0)
    indicators = [r'$V/Ni=%.2f$' % (V_Ni_Ratio, ), r'$OC/EC=%.2f$' % (OC_EC_Ratio, )]
    textstr = '\n'.join(indicators) 

    ax4.set_title("Factor Profile - Run " + str(optimal_run) + " - " + factor)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig.text(0.25, 0.75, textstr, fontsize=12, verticalalignment='top', bbox=props)   
    #fig.text(0.75, 0.5, fontsize=12, verticalalignment='top', bbox=props)   
    fig.legend(fontsize = 12, loc = 'upper right', borderaxespad = 5)
    fig.autofmt_xdate()

    plt.draw()
    figure = plt.gcf()
    figure.set_size_inches(14,5)
    #fig.savefig(path.join(outpath, f"factor_profile_{factor}.png"), bbox_inches = "tight", dpi = 100)

### FACTOR CONTRIBUTIONS ###
contributions = contributions.groupby(pd.Grouper(key = 'Date', freq= 'M'))
dfs = [group for _,group in contributions]

# months 
first_date = dfs[0]["Date"].iloc[0]
last_date = dfs[-1]["Date"].iloc[0]
months_list = pd.date_range(str(first_date)[:7], last_date, freq = 'MS',closed="left").strftime("%Y-%b").tolist()

# loop through factors_list to generate factor contributions for each factor
for factor in factors_list:
    months_dict = {}
    for i in range(len(months_list)):
        months_dict[months_list[i]] = np.asarray(dfs[i][factor])

    months_data = pd.DataFrame.from_dict(months_dict, orient='index')
    months_data = months_data.transpose()

    fig, ax6 = plt.subplots()
    meanprops = dict(marker = "s", markerfacecolor = "r", markeredgecolor = "black", ms = 5) # mean dot
    flierprops = dict(marker = "o")
    sns.boxplot(data = months_data, ax = ax6, showmeans = True, color = "skyblue", meanprops = meanprops, flierprops = flierprops)
    sns.despine()

    #ax6.set_xlabel("Months")
    ax6.set_ylabel(u"$PM_{2.5}$ (\u03bcg/$m^3$)")
    ax6.set_title("Factor Contributions - Run " + str(optimal_run) + " - " + factor)
    plt.xticks(rotation=45)

    plt.draw()
    figure = plt.gcf()
    figure.set_size_inches(14,5)
    #fig.savefig(path.join(outpath, f"factor_contributions_{factor}.png"), bbox_inches = "tight", dpi = 100)

### Linear Regression Plot ###
conc_data_lr = conc_data

# split into est_conc (other species) and est conc (PM2.5)
est_conc_others = est_conc[:240]
est_conc_PM25 = est_conc[244:]

# Calculate estimated concentrations 
species_conc_data = conc_data_lr[conc_data_lr["Species"] == user_input_species]
est_species = np.array(est_conc_others.iloc[:,1:]) * np.array(species_conc_data.iloc[:,1:])
est_species = np.where(est_species >= 0, est_species, 0)
est_species = np.sum(est_species, axis = 1)
est_conc_others["Est. Species"] = pd.Series(est_species)

# Combine est_conc and observed conc into one df 
observed_conc_species = observed_conc.loc[:, ["Date",user_input_species]]
conc_df = pd.merge(est_conc_others, observed_conc_species, on = "Date", how = "left")
conc_df["Est. Species"] = conc_df["Est. Species"].astype(float)
x = conc_df["Est. Species"]
y = conc_df[user_input_species]
print(conc_df)

# Find out linear regression equation & r2
slope, intercept, r, p, std_err = stats.linregress(x,y)
regression_line = slope*x + intercept
r2 = r**2

# plot linear regression
fig, ax7 = plt.subplots()

sns.regplot(data = conc_df, ax = ax7, x = "Est. Species", y = user_input_species, fit_reg = True, ci = 95, n_boot = 1000)

ax7.set_title(f"{user_input_species}")
ax7.set_xlabel("Observed Concentrations")
ax7.set_ylabel("Predicted Concentrations")

eqn = 'y = {:.2f}x + {:.2f}'.format(slope, intercept)
p_indicator = ("< 0.05") if (p < 0.05) else ("> 0.05")
labels = [eqn, '$r^2$ = {:.2f}'.format(r2), f"p {p_indicator}"]
textstr = '\n'.join(labels)

fig.text(0.25, 0.75, textstr, fontsize=12, verticalalignment='top') 

plt.draw()
figure = plt.gcf()
figure.set_size_inches(14,5)
#fig.savefig(path.join(outpath, f"linear_regression_{user_input_species}.png"), bbox_inches = "tight", dpi = 100)

plt.show()








