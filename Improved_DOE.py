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

np.set_printoptions(legacy='1.25')

class DOE:
    def __init__(self, doe_file, x_cols=None, y_col=None,levels_list=None, all_response_list=None, full_df=None, interaction_cols=None):
        self.doe_file = doe_file
        self.df = pd.read_csv(self.doe_file)
        self.x_cols = x_cols
        self.y_col = y_col
        self.df.columns = self.df.columns.str.strip()
        self.levels_list = levels_list
        self.all_response_list = all_response_list
        self.full_df = full_df
        self.interaction_cols = interaction_cols

        print(self.df)
        print('='*100)

        # assume that final col is response, rest are variables
        col_no = len(self.df.columns)
        x_col = []
        for i in range(col_no - 1):
            x_col.append(self.df.columns[i])

        self.x_cols = x_col
        self.y_col = self.df.columns[-1]

        # need to convert high, med. low to +1, 0, -1
        for col in self.x_cols:

            max_val = self.df[col].max()
            min_val = self.df[col].min()

            mapping = {
                max_val: 1,
                min_val: -1
            }

            self.df[col] = self.df[col].map(mapping).fillna(0)

        print('Converted DF:')
        print(self.df)
        print('=' * 100)


    def interactions(self):
        # get interactions which may be feasible, e.g.,
        # Temp. x Catalyst_mol%
        # give choice of automatic vs manual

        combos = self.x_cols
        all_combos = list(combinations(combos, 2))

        manual = input('Do you want automatic (A) or manual (M) interaction selection? ')

        if manual == 'M':
            print(
                f'Based on the following {len(all_combos)} combinations, choose which to keep by inputting (Y) or (N) for each combination:')
        if manual == 'A':
            print(f'{len(all_combos)} combinations will be automatically selected.')

        combo_list = []

        for combo in all_combos:
            print(combo)

            if manual == 'A':
                combo_list = all_combos

            if manual == 'M':
                chosen = input('')
                if chosen == 'Y':
                    combo_list.append(combo)
                else:
                    pass

        if len(combo_list) == 0:
            print('No combinations selected.')
            self.full_df = self.df
            pass
        else:

            if len(combo_list) > 0:
                interactions_df = pd.DataFrame()
                self.interactions_df = interactions_df
                self.new_cols = []

                for combo in combo_list:
                    # unpacks tuple into list -> (A, B) => [A, B]
                    curr_combo = [*combo]

                    col1, col2 = curr_combo
                    new_col_name = col1 + ' x ' + col2
                    self.new_cols.append(new_col_name)

                    self.interactions_df[new_col_name] = self.df[col1] * self.df[col2]

                self.full_df = pd.concat([self.df, self.interactions_df], axis=1)
                updated_order = [col for col in self.full_df if col != self.y_col] + [self.y_col]
                self.full_df = self.full_df[updated_order]

                # self.full_df.to_csv('TEST.csv', index=False)
                print('=' * 100)

                col_no = len(self.full_df.columns)
                x_col = []
                for i in range(col_no - 1):
                    x_col.append(self.full_df.columns[i])

                self.interaction_cols = x_col

                return self.full_df

    def regression_model(self,alpha = 0.05):

        for n in range(1, 3):
            poly = PolynomialFeatures(degree=n, include_bias=False)

            if self.interaction_cols is None:
                print(self.x_cols)
                print(self.full_df)

                x_poly = poly.fit_transform(self.full_df[self.x_cols])
                feature_names = poly.get_feature_names_out(self.x_cols)
            else:
                x_poly = poly.fit_transform(self.full_df[self.interaction_cols])
                feature_names = poly.get_feature_names_out(self.interaction_cols)

            x_poly_df = pd.DataFrame(x_poly, columns=feature_names, index=self.full_df.index)
            x_poly_with_intercept = sm.add_constant(x_poly_df, has_constant='add')

            model = LinearRegression()
            model.fit(x_poly_df, self.full_df[self.y_col])

            print(f'Polynomial degree: {n}')

            osl_model = sm.OLS(self.full_df[self.y_col], x_poly_with_intercept).fit()
            print(osl_model.summary())

            all_names = ['Intercept'] + list(feature_names)
            p_values = osl_model.pvalues.values
            coefs = osl_model.params.values

            results_df = pd.DataFrame({
                'Feature': all_names,
                'Coefficient': np.round(coefs, 4),
                'P-value': np.round(p_values, 4)
            })
            results_df['Significant'] = results_df['P-value'] < alpha
            results_df = results_df.sort_values(by='Coefficient', ascending=False)

            temp_df = results_df[results_df['Feature'] != 'Intercept']

            colors = []
            for sig in temp_df['Significant']:
                if sig:
                    colors.append('#1f77b4')
                else:
                    colors.append('#f5424e')

            fig, ax = plt.subplots(figsize=(10, 10))
            bars = ax.bar(temp_df['Feature'], temp_df['Coefficient'], color=colors)
            plt.ylabel('Coefficients')
            plt.axhline(y=0, color='k', lw=0.5)
            plt.xticks(rotation=90)
            plt.title(f'Coefficients (blue = p < {alpha}, red = not significant)')
            plt.tight_layout()

            for rect, p_val in zip(bars, temp_df['P-value']):
                height = rect.get_height()
                va = 'bottom' if height >= 0 else 'top'
                plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'p={p_val:.3f}', ha='center', va=va)

            plt.show()
            print('=' * 100)

            # need to now look at a residuals plot
            y_pred = model.predict(x_poly_df)
            residuals = y_pred - self.df[self.y_col]

            plt.scatter(y_pred, residuals)
            plt.axhline(y=0, color='k', lw=0.5)
            plt.xlabel('Predicted values')
            plt.ylabel('Residuals')
            plt.title(f'Residuals vs Fitted | Order = {n}')
            plt.show()

        print('Based on the information provided, do you wish to eliminate any factors/interactions?:')
        remove_input = input('Input (Y) to remove factors/interactions or, (N) to continue: ')
        remain_list = []
        if remove_input == 'Y':
            if self.interaction_cols is not None:
                for col in self.interaction_cols:
                    print(col)
                    add = input('')
                    if add == 'Y':
                        pass
                    else:
                        remain_list.append(col)
            else:
                for col in self.x_cols:
                    print(col)
                    add = input('')
                    if add == 'Y':
                        pass
                    else:
                        remain_list.append(col)
        print('Remaining factors:')
        print(remain_list)

    def main_effects(self):
        # get mean response at each level (y) and plot against (-1,0,1)

        all_response_list = []
        response_name = []
        levels_list = []

        if hasattr(self, 'full_df'):
            for col in self.full_df.columns:
                if col == self.y_col:
                    pass
                else:
                    uniques = pd.unique(self.full_df[col])
                    no_levels = len(uniques)

                    levels_list.append(no_levels)

                    max_response = self.full_df[self.full_df[col] == 1][self.y_col].mean()
                    min_response = self.full_df[self.full_df[col] == -1][self.y_col].mean()
                    if no_levels == 3:
                        mid_response = self.full_df[self.full_df[col] == 0][self.y_col].mean()
                        all_response_list.append([min_response, mid_response, max_response])
                    else:
                        all_response_list.append([min_response, max_response])

                    response_name.append(col)
        else:
            for col in self.x_cols:
                if col == self.y_col:
                    pass
                else:
                    uniques = pd.unique(self.df[col])
                    no_levels = len(uniques)

                    levels_list.append(no_levels)

                    max_response = self.df[self.df[col] == 1][self.y_col].mean()
                    min_response = self.df[self.df[col] == -1][self.y_col].mean()
                    if no_levels == 3:
                        mid_response = self.df[self.df[col] == 0][self.y_col].mean()
                        all_response_list.append([min_response, mid_response, max_response])
                    else:
                        all_response_list.append([min_response, max_response])

                    response_name.append(col)
        count=0
        for responses in all_response_list:
            if levels_list[count] == 3:
                xs = [-1, 0, 1]
            if levels_list[count] == 2:
                xs = [-1, 1]
            plt.plot(xs, responses)
            plt.title(f'Main Effects: {response_name[count]}')
            count += 1
            plt.axhline(self.df[self.y_col].max(), color='r', alpha=0.5, linestyle='--')
            plt.axhline(self.df[self.y_col].min(), color='r', alpha=0.5, linestyle='--')
            plt.xlabel('Levels')
            plt.xticks(xs)
            plt.ylabel(self.y_col)
            plt.show()

        self.levels_list = levels_list
        self.all_response_list = all_response_list


    def interaction_plots(self):
        # need to make pairs between two factors, e.g., if you have A, B, C, D ...
        # A+B, A+C, A+D, B+C, B+D, C+D
        # from each factor, need to get its mean response at -1 and +1

        factor_combos = list(combinations(self.x_cols, 2))
        print('Combination of factors:')
        print(factor_combos)

        interactions_list = []
        # in the order of [A-1, B-1], [A+1, B-1], [A-1, B+1], [A+1, B+1]

        xs = [-1,1]
        for factor_a, factor_b in factor_combos:

            # combos are in order of interactions list
            # starting w/ mins of A and then maxs of A
            combo_1 = self.df[(self.df[factor_a] == -1) & (self.df[factor_b] == -1)][self.y_col].mean()
            combo_2 = self.df[(self.df[factor_a] == -1) & (self.df[factor_b] == 1)][self.y_col].mean()
            combo_3 = self.df[(self.df[factor_a] == 1) & (self.df[factor_b] == -1)][self.y_col].mean()
            combo_4 = self.df[(self.df[factor_a] == 1) & (self.df[factor_b] == 1)][self.y_col].mean()

            plt.plot(xs, [combo_1, combo_2], label=f'{factor_a} = -1')
            plt.plot(xs, [combo_3, combo_4], label=f'{factor_a} = 1')
            plt.xlabel(factor_b)
            plt.ylabel(self.y_col)
            plt.legend()
            plt.xticks(xs)
            plt.title(f'Interaction: {factor_a} x {factor_b}')
            plt.axhline(self.df[self.y_col].max(), color='r', alpha=0.5, linestyle='--')
            plt.axhline(self.df[self.y_col].min(), color='r', alpha=0.5, linestyle='--')
            plt.show()




t = DOE('reaction_DOE.csv')
t.interactions()
t.regression_model(alpha=0.05)
t.main_effects()
t.interaction_plots()
