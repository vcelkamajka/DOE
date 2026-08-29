import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
import pandas as pd
import statsmodels.api as sm
from itertools import combinations, permutations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import f_oneway
from scipy.stats import shapiro
from matplotlib.ticker import FormatStrFormatter
from statsmodels.formula.api import ols
import re

def clean_col(col):
    return re.sub(r'\W+', '_', col).strip('_')

# full factorial: all combinations
# fractional factorial: aliased factors, less runs -> less accurate, removes high level interactions
# e.g., if we guess that D is confounding w/ A, B and C -> D = A x B x C
# so if A = -1, B = 1 and C = 1, D = -1
# Central Composite Design (CCD) : add axial points at typically +- 1.414
# Box-Behnken : combinations kept within +- 1

# helps make the necessary confounded factors, i.e., AB, BC, ABC, ABCD... (currently only goes to 4)
def get_factor_pairs(factor_list):
    return (list(combinations(factor_list, 2)) + list(combinations(factor_list, 3)) + list(
        combinations(factor_list, 4)))


class DOE:
    def __init__(self, file_name, df=None, x_cols=None, y_col=None):
        self.file_name = file_name
        self.x_cols = x_cols
        self.y_col = y_col
        self.df = pd.read_csv(file_name)

        self.x_cols = []
        self.y_col = []

        print(self.df)

        for col in self.df.columns:
            self.x_cols.append(col)
        self.x_cols.pop(-1)
        self.y_col = self.df.columns[-1]

        print('=' * 50)
        print(f'X-features: {self.x_cols}')
        print(f'Y-feature: {self.y_col}')
        print('=' * 50)
        print('Converted DF:')
        # need to standardise values to 1 and -1
        # get max, min and else values (+1, -1 and 0)
        # once values established must map values to +1, -1 and 0

        for col in self.x_cols:
            max_val = max(self.df[col])
            min_val = min(self.df[col])
            map_dict = {max_val: 1, min_val: -1}

            self.df[col] = self.df[col].map(map_dict).fillna(0)

        print(self.df)
        print('=' * 50)

    def stats(self):

        temp = self.df.copy()
        temp = self.df.rename(columns={col: clean_col(col) for col in temp.columns})
        temp_y_col = clean_col(self.y_col)
        temp_x_cols = [clean_col(c) for c in self.x_cols]

        formula = f"{temp_y_col} ~ {' + '.join(temp_x_cols)}"
        anova_model = ols(formula, data=temp).fit()
        anova_table = sm.stats.anova_lm(anova_model, typ=2)
        print(anova_table)

    def add_interactions(self):
        # e.g., if you have factors A, B, C -> need to construct factors AB, AC, BC and ABC
        # if A = -1 and B = -1, AB = +1
        factor_list = self.x_cols
        start_pair = get_factor_pairs(factor_list)

        pre_join_list = []
        new_factor_list = []
        for factors in start_pair:
            pairs_list = []
            for pairs in factors:
                pairs_list.append(pairs)
                if len(factors) == 2 and len(pairs_list) == 2:
                    res = pairs_list[0] + ' x ' + pairs_list[1]
                    new_factor_list.append(res)
                if len(factors) == 3 and len(pairs_list) == 3:
                    res = pairs_list[0] + ' x ' + pairs_list[1] + ' x ' + pairs_list[2]
                    new_factor_list.append(res)
                if len(factors) == 4 and len(pairs_list) == 4:
                    res = pairs_list[0] + ' x ' + pairs_list[1] + ' x ' + pairs_list[2] + ' x ' + pairs_list[3]
                    new_factor_list.append(res)
            pre_join_list.append(list(factors))  # this list is the list of each combination

        print(f'Following new factors have been made in the form AB, ABC, etc., : \n{new_factor_list}')

        for pair in pre_join_list:
            if len(pair) == 2:
                # col_set_1 = pair[0], pair[1]
                # print('col_set_1: ', col_set_1)
                # I AM THE GOAT FOR FIGURING TS OUT OMFD
                self.df[pair[0] + ' x ' + pair[1]] = self.df[pair[0]] * self.df[pair[1]]

            if len(pair) == 3:
                # col_set_2 = pair[0], pair[1], pair[2]
                # print('col_set_2: ', col_set_2)
                self.df[pair[0] + ' x ' + pair[1] + ' x ' + pair[2]] = self.df[pair[0]] * self.df[pair[1]] * self.df[
                    pair[2]]

            if len(pair) == 4:
                # print(pair[0], pair[1], pair[2], pair[3])
                self.df[pair[0] + ' x ' + pair[1] + ' x ' + pair[2] + ' x ' + pair[3]] = self.df[pair[0]] * self.df[
                    pair[1]] * self.df[
                                                                                             pair[2]] * self.df[pair[3]]

        # for new_factor in new_factor_list:
        #     self.df[new_factor] = self.df.Temperature * self.df.Stir_Rate

        len_of_cols = len(self.df.columns)
        temp_y = self.df.pop(self.y_col)
        self.df.insert(len_of_cols - 1, temp_y.name, temp_y)
        print(self.df)
        self.df.to_csv('factors.csv')

        print(
            f'Of the following, choose what interactions to keep, including too many will lead to incomprehensible results!\n {new_factor_list}')
        print(
            "Please input your selected features one-by-one, i.e., for option '1', input 0 and click enter. To end please input 'END'.")

        keep_list = []
        while True:
            keep = input('')
            keep_list.append(keep)
            if keep == 'END':
                keep_list.pop(-1)
                break

        target_col = self.y_col

        for n in keep_list:
            col_to_add = new_factor_list[int(n)]
            self.x_cols.append(col_to_add)

        print(self.x_cols)

        self.df = self.df[self.x_cols + [target_col]]
        self.y_col = target_col

        self.y_col = self.df.columns[-1]

        print('=' * 50)

    def main_effects(self):

        for col in self.x_cols:
            max_ys = []
            min_ys = []
            zero_ys = []

            for n in range(len(self.df[col])):
                if self.df[col][n] == 1:
                    max_ys.append(self.df[self.y_col][n])
                if self.df[col][n] == -1:
                    min_ys.append(self.df[self.y_col][n])
                if self.df[col][n] == 0:
                    zero_ys.append(self.df[self.y_col][n])

            if len(zero_ys) > 0:
                xs = [-1, 0, 1]
                ys = [np.mean(min_ys), np.mean(zero_ys), np.mean(max_ys)]
            else:
                xs = [-1, 1]
                ys = [np.mean(min_ys), np.mean(max_ys)]

            fig, ax = plt.subplots()
            plt.plot(xs, ys)
            plt.title('Main Effects Plot')
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            plt.xticks([-1, 0, 1])
            plt.xlabel(self.df[col].name)
            plt.ylabel(self.df[self.y_col].name)
            plt.show()

            ys = []

    def interaction_plots(self):
        # get the equation from coefficients and plot with alternating
        pass

    def get_trend(self, n, alpha=0.05, x_vals=None, y_vals=None, model=None,x_poly_df=None):
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.model = model

        self.x_vals = self.df[self.x_cols]
        self.y_vals = self.df[self.y_col]

        poly = PolynomialFeatures(n, include_bias=False)
        self.X_poly = poly.fit_transform(self.x_vals)
        x_poly = poly.fit_transform(self.x_vals)
        self.model = LinearRegression()
        self.model.fit(x_poly, self.y_vals)
        y_pred = self.model.predict(x_poly)
        intercept = self.model.intercept_

        x_sm = sm.add_constant(x_poly, has_constant='add')
        self.osl_model = sm.OLS(self.y_vals, x_sm).fit()
        print(self.osl_model.summary())

        with open("OLS_summary.csv", "w") as f:
            f.write(self.osl_model.summary().as_csv())

        # model_coefficients = self.model.coef_
        feature_names = poly.get_feature_names_out(self.x_vals.columns)
        self.x_poly_df = pd.DataFrame(x_poly,columns=feature_names)
        all_names = ['Intercept'] + list(feature_names)

        p_values = self.osl_model.pvalues.values
        coefs = self.osl_model.params.values

        ci = self.osl_model.conf_int()

        results_df = pd.DataFrame({
            'Feature': all_names,
            'Coefficient': np.round(coefs, 4),
            'P-value': np.round(p_values, 4),
            f'{alpha / 2} CI': np.round(ci[0], 4),
            f'{1 - alpha / 2} CI': np.round(ci[1], 4),
        })

        results_df['Significant'] = results_df['P-value'] < alpha
        results_df = results_df.sort_values(by='Coefficient', ascending=False)
        print('Results dataframe:')
        print(results_df)

        results_df.to_csv('results.csv')

        colors = []
        for sig in results_df['Significant']:
            if sig:
                colors.append('#1f77b4')
            else:
                colors.append('#f5424e')

        fig, ax = plt.subplots(figsize=(10, 10))
        temp = results_df.drop('const')
        plt.bar(temp['Feature'], temp['Coefficient'], color=colors)
        plt.xlabel('Features')
        plt.ylabel('Coefficients')
        plt.axhline(y=0, color='k', lw=0.5)
        plt.xticks(rotation=90)
        plt.title(f'Coefficients (blue = p < {alpha}, red = not significant)')
        plt.tight_layout()
        plt.show()

        r2 = r2_score(self.y_vals, y_pred)
        mse = mean_squared_error(self.y_vals, y_pred)
        print('=' * 50)
        print(f'R²: {r2:.2f}')
        print(f'Adjusted R²: {self.osl_model.rsquared_adj:.3f}')
        print(f'MSE: {mse:.2f}')
        print('=' * 50)

        vif_data = pd.DataFrame()
        vif_data['Feature'] = self.x_cols
        vif_data['VIF'] = [variance_inflation_factor(self.x_vals.values, i) for i in range(len(self.x_cols))]
        print('VIF of 1 = no multicollinearity, VIF of 1-5 = moderate collinearity, VIF >5-10 = Serious collinearity!')
        print(vif_data)

        print('=' * 50)

        self.results_df = results_df
        return self.results_df

    def eliminate_factors(self, sig_factors=None,remaining_factors=3):
        self.remaining_factors = remaining_factors
        self.sig_factors = sig_factors
        # eliminate factors to a user defined level - default is 3

        sig_factors = self.results_df.loc[self.results_df['Significant'] == True, 'Feature']
        sig_factors = sig_factors.drop('const')
        display(sig_factors)

        if len(sig_factors) > remaining_factors:
            print(f'Excess factors remain - need to remove {len(sig_factors) - remaining_factors}!')
        if len(sig_factors) < remaining_factors:
            print(f'Satisfactory amount of factors remain.')

        selection_list = []
        for features in sig_factors:
            selection = self.results_df.loc[self.results_df['Feature'] == features]
            selection_list.append(selection)
            display(selection)

        temp = self.results_df
        temp = temp.drop('const')
        temp['Coefficient'] = np.abs(temp['Coefficient'])

        manual_query = input('Do you wish to automatically eliminate factors based on ascending coefficient value? (Y/N): ')

        if manual_query == 'N':
            count = 0
            to_drop = []
            for n in range(len(sig_factors) - remaining_factors):
                drop = input(
                    f"Based on the results shown above, please drop {len(sig_factors) - remaining_factors + count} factors by inputting the axis value, e.g., x1, click enter and if required input another afterwards e.g., x2... ")
                count -= 1
                to_drop.append(drop)

            for drop in to_drop:
                sig_factors = sig_factors.drop(drop)
            print('Remaining features:')
            display(sig_factors)
            self.sig_factors = sig_factors

        if manual_query == 'Y':
            temp = temp.sort_values(by='Coefficient', ascending=False)
            index_range = len(sig_factors) - (remaining_factors+1)
            temp.drop(temp.index[index_range:], inplace=True)
            print(temp)
            sig_factors = temp['Feature']
            self.sig_factors = sig_factors

        print('=' * 50)

        x_reduced = self.x_poly_df[sig_factors]

        x_reduced_sm = sm.add_constant(x_reduced,has_constant='add')
        reduced_model = sm.OLS(self.y_vals, x_reduced_sm).fit()
        print(reduced_model.summary())

        print(f'Full model R²: {self.osl_model.rsquared:.2f}, Adj R²: {self.osl_model.rsquared_adj:.3f}')
        print(f'Reduced model R²: {reduced_model.rsquared:.2f}, Adj R²: {reduced_model.rsquared_adj:.3f}')

        with open("Adjusted_OLS_summary.csv", "w") as f:
            f.write(self.osl_model.summary().as_csv())

        self.x_cols = []
        for n in self.sig_factors:
            self.x_cols.append(n)


    def RSM(self):

        x_coefs_df = pd.DataFrame()
        x_coefs_df = self.results_df[self.results_df['Feature'].isin(self.x_cols)]
        x_coefs_df = x_coefs_df.drop(['P-value','0.025 CI','0.975 CI', 'Significant'], axis='columns')
        print(x_coefs_df)
        temp = x_coefs_df

        fig, ax = plt.subplots(figsize=(10, 10), kwargs={'projection':'3d'})

        if self.remaining_factors == 3:

            x = np.linspace()
            y = np.linspace()
            x,y = np.meshgrid(x,y)

            z =

            ax.plot_surface(x,y,z, cmap='inferno',antialiased=False)

            plt.show()




# need to construct 2D &/or 3D RSM of results and maximise/minimise


t = DOE('reaction_DOE.csv')
t.add_interactions()
t.stats()
t.main_effects()
t.interaction_plots()
t.get_trend(2)
t.eliminate_factors(remaining_factors=3)
t.RSM()
