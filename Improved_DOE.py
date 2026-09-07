import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import combinations, permutations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import f_oneway
from scipy.stats import shapiro

np.set_printoptions(legacy='1.25')


class DOE:
    def __init__(self, doe_file, x_cols=None, y_col=None, levels_list=None,
                 all_response_list=None, full_df=None, interaction_cols=None):
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
        print('=' * 100)

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
            self.interaction_cols = None
        else:
            interactions_df = pd.DataFrame()
            self.interactions_df = interactions_df
            self.new_cols = []

            for combo in combo_list:
                curr_combo = [*combo]
                col1, col2 = curr_combo
                new_col_name = col1 + ' x ' + col2
                self.new_cols.append(new_col_name)

                self.interactions_df[new_col_name] = self.df[col1] * self.df[col2]

            self.full_df = pd.concat([self.df, self.interactions_df], axis=1)
            updated_order = [col for col in self.full_df if col != self.y_col] + [self.y_col]
            self.full_df = self.full_df[updated_order]

            print('=' * 100)

            col_no = len(self.full_df.columns)
            x_col = []
            for i in range(col_no - 1):
                x_col.append(self.full_df.columns[i])

            self.interaction_cols = x_col

            return self.full_df

    def _compute_vif(self, x_poly_df):
        """VIF computed on the ACTUAL features fed into the model (x_poly_df),
        not the raw pre-expansion factors - otherwise the numbers describe a
        different feature set than the one actually fit."""
        vif_data = pd.DataFrame()
        vif_data['Feature'] = x_poly_df.columns
        vif_data['VIF'] = [variance_inflation_factor(x_poly_df.values, i)
                            for i in range(x_poly_df.shape[1])]
        print('VIF ~1 = no multicollinearity, 1-5 = moderate, >5-10 = serious concern')
        print(vif_data)
        print('=' * 100)
        return vif_data

    def _plot_pareto_coefficients(self, results_df, alpha, order_label=''):
        """Sorted by ABSOLUTE coefficient value (largest effect first),
        which is what a Pareto-style ranking actually requires - sorting on
        the signed value hides large negative effects near the 'unimportant' end."""
        temp_df = results_df[results_df['Feature'] != 'Intercept'].copy()
        temp_df['AbsCoefficient'] = temp_df['Coefficient'].abs()
        temp_df = temp_df.sort_values(by='AbsCoefficient', ascending=False)

        colors = ['#1f77b4' if sig else '#f5424e' for sig in temp_df['Significant']]

        fig, ax = plt.subplots(figsize=(10, 10))
        bars = ax.bar(temp_df['Feature'], temp_df['Coefficient'], color=colors)
        plt.ylabel('Coefficients')
        plt.axhline(y=0, color='k', lw=0.5)
        plt.xticks(rotation=90)
        plt.title(f'Pareto of Effects {order_label} (blue = p < {alpha}, red = not significant)')
        plt.tight_layout()

        for rect, p_val in zip(bars, temp_df['P-value']):
            height = rect.get_height()
            va = 'bottom' if height >= 0 else 'top'
            plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'p={p_val:.3f}',
                      ha='center', va=va)

        plt.show()

    def regression_model(self, alpha=0.05):

        last_results_df = None

        for n in range(1, 3):
            poly = PolynomialFeatures(degree=n, include_bias=False)

            if self.interaction_cols is None:
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
            print(f'Adjusted R^2: {osl_model.rsquared_adj:.4f} | Residual df: {osl_model.df_resid}')

            all_names = ['Intercept'] + list(feature_names)
            p_values = osl_model.pvalues.values
            coefs = osl_model.params.values

            results_df = pd.DataFrame({
                'Feature': all_names,
                'Coefficient': np.round(coefs, 4),
                'P-value': np.round(p_values, 4)
            })
            results_df['Significant'] = results_df['P-value'] < alpha
            last_results_df = results_df

            self._plot_pareto_coefficients(results_df, alpha, order_label=f'| Order = {n}')
            print('=' * 100)

            self._compute_vif(x_poly_df)

            y_pred = model.predict(x_poly_df)
            residuals = y_pred - self.full_df[self.y_col]

            plt.scatter(y_pred, residuals)
            plt.axhline(y=0, color='k', lw=0.5)
            plt.xlabel('Predicted values')
            plt.ylabel('Residuals')
            plt.title(f'Residuals vs Fitted | Order = {n}')
            plt.show()

        # keep a lookup of which base factors are involved in a significant
        # interaction, so elimination can warn about breaking hierarchy
        sig_terms = set(last_results_df.loc[last_results_df['Significant'], 'Feature'])
        factors_in_sig_interactions = set()
        for term in sig_terms:
            if ' x ' in term or ' ' in term:
                for part in term.replace(' x ', ' ').split(' '):
                    if part in self.x_cols or (self.interaction_cols and part in self.interaction_cols):
                        factors_in_sig_interactions.add(part)

        print('Based on the information provided, do you wish to eliminate any factors/interactions?:')
        remove_input = input('Input (Y) to remove factors/interactions or, (N) to continue: ')
        remain_list = []
        source_cols = self.interaction_cols if self.interaction_cols is not None else self.x_cols

        if remove_input == 'Y':
            for col in source_cols:
                # show significance context so the decision isn't made blind
                row = last_results_df[last_results_df['Feature'].str.contains(
                    col.replace(' x ', ' '), regex=False)]
                sig_note = 'SIGNIFICANT' if not row.empty and row['Significant'].any() else 'not significant'
                hierarchy_note = ''
                if col in factors_in_sig_interactions:
                    hierarchy_note = '  [WARNING: part of a significant interaction - dropping breaks hierarchy]'
                print(f'{col}  ({sig_note}){hierarchy_note}')

                keep = input('Keep this factor/interaction? (Y = keep, N = drop): ')
                if keep == 'Y':
                    remain_list.append(col)
                # anything other than 'Y' drops it
        if remove_input == 'N':
            remain_list.extend(source_cols)

        print('Remaining factors:')
        print(remain_list)

        # -------------------------- REPEAT ON REDUCED FEATURE SET

        for n in range(1, 3):
            poly_2 = PolynomialFeatures(degree=n, include_bias=False)

            x_poly_2 = poly_2.fit_transform(self.full_df[remain_list])
            feature_names_2 = poly_2.get_feature_names_out(remain_list)

            x_poly_df_2 = pd.DataFrame(x_poly_2, columns=feature_names_2, index=self.full_df.index)
            x_poly_with_intercept_2 = sm.add_constant(x_poly_df_2, has_constant='add')

            model_2 = LinearRegression()
            model_2.fit(x_poly_df_2, self.full_df[self.y_col])

            print(f'Polynomial degree: {n}')

            osl_model_2 = sm.OLS(self.full_df[self.y_col], x_poly_with_intercept_2).fit()
            print(osl_model_2.summary())
            print(f'Adjusted R^2: {osl_model_2.rsquared_adj:.4f} | Residual df: {osl_model_2.df_resid}')

            all_names_2 = ['Intercept'] + list(feature_names_2)
            p_values_2 = osl_model_2.pvalues.values
            coefs_2 = osl_model_2.params.values

            results_df_2 = pd.DataFrame({
                'Feature': all_names_2,
                'Coefficient': np.round(coefs_2, 4),
                'P-value': np.round(p_values_2, 4)
            })
            results_df_2['Significant'] = results_df_2['P-value'] < alpha

            self._plot_pareto_coefficients(results_df_2, alpha, order_label=f'(Reduced) | Order = {n}')
            print('=' * 100)

            self._compute_vif(x_poly_df_2)

            y_pred_2 = model_2.predict(x_poly_df_2)
            residuals_2 = y_pred_2 - self.full_df[self.y_col]

            plt.scatter(y_pred_2, residuals_2)
            plt.axhline(y=0, color='k', lw=0.5)
            plt.xlabel('Predicted values')
            plt.ylabel('Residuals')
            plt.title(f'Residuals vs Fitted (Reduced) | Order = {n}')
            plt.show()

            self.final_results = results_df_2  # used later to guide RSM axis selection

        print('=' * 100)
        print('Based on the two models ran - select a model to use for RSM:')

        model_pick = input('If you want to use model 1 (full) input 1, model 2 (reduced) input 2: ')
        regression_pick = input('If you want to use order 1 input 1, order 2 input 2: ')

        self.remain_list = remain_list

        if model_pick == '1':
            self.choice = 1
        if model_pick == '2':
            self.choice = 2
        if regression_pick == '1':
            self.order = 1
        if regression_pick == '2':
            self.order = 2

        print('=' * 100)

    def main_effects(self):
        # get mean response at each level (y) and plot against (-1,0,1)

        all_response_list = []
        response_name = []
        levels_list = []

        source_df = self.full_df if hasattr(self, 'full_df') and self.full_df is not None else self.df
        cols_to_check = source_df.columns if hasattr(self, 'full_df') and self.full_df is not None else self.x_cols

        for col in cols_to_check:
            if col == self.y_col:
                continue
            uniques = pd.unique(source_df[col])
            no_levels = len(uniques)
            levels_list.append(no_levels)

            max_response = source_df[source_df[col] == 1][self.y_col].mean()
            min_response = source_df[source_df[col] == -1][self.y_col].mean()
            if no_levels == 3:
                mid_response = source_df[source_df[col] == 0][self.y_col].mean()
                all_response_list.append([min_response, mid_response, max_response])
            else:
                all_response_list.append([min_response, max_response])

            response_name.append(col)

        count = 0
        for responses in all_response_list:
            xs = [-1, 0, 1] if levels_list[count] == 3 else [-1, 1]
            plt.plot(xs, responses)
            plt.title(f'Main Effects: {response_name[count]}')
            plt.axhline(self.df[self.y_col].max(), color='r', alpha=0.5, linestyle='--')
            plt.axhline(self.df[self.y_col].min(), color='r', alpha=0.5, linestyle='--')
            plt.xlabel('Levels')
            plt.xticks(xs)
            plt.ylabel(self.y_col)
            plt.show()
            count += 1

        self.levels_list = levels_list
        self.all_response_list = all_response_list

    def interaction_plots(self):
        # need to make pairs between two factors, e.g., if you have A, B, C, D ...
        factor_combos = list(combinations(self.x_cols, 2))
        print('Combination of factors:')
        print(factor_combos)
        print('=' * 100)

        xs = [-1, 1]
        for factor_a, factor_b in factor_combos:
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

    def RSM(self):

        if self.choice == 1:
            poly = PolynomialFeatures(degree=self.order, include_bias=False)
            x_poly = poly.fit_transform(self.full_df[self.x_cols])
            feature_names = poly.get_feature_names_out(self.x_cols)
            x_poly_df = pd.DataFrame(x_poly, columns=feature_names, index=self.full_df.index)
            x_poly_with_intercept = sm.add_constant(x_poly_df, has_constant='add')
            model = LinearRegression()
            model.fit(x_poly_df, self.full_df[self.y_col])
            x_cols = self.x_cols

        if self.choice == 2:
            poly = PolynomialFeatures(degree=self.order, include_bias=False)
            x_poly = poly.fit_transform(self.full_df[self.remain_list])
            feature_names = poly.get_feature_names_out(self.remain_list)
            x_poly_df = pd.DataFrame(x_poly, columns=feature_names, index=self.full_df.index)
            x_poly_with_intercept = sm.add_constant(x_poly_df, has_constant='add')
            model = LinearRegression()
            model.fit(x_poly_df, self.full_df[self.y_col])
            x_cols = self.remain_list

        print('Based on the factors below, input the list index to select the two most '
              'critical factors, ensure to press enter between each entry.')

        for i, factor in enumerate(self.x_cols):
            note = ''
            if hasattr(self, 'final_results'):
                involved = self.final_results[
                    (self.final_results['Feature'].str.contains(factor, regex=False)) &
                    (self.final_results['Significant'])
                ]
                if not involved.empty:
                    sig_terms = ', '.join(involved['Feature'].tolist())
                    note = f'  <- involved in significant term(s): {sig_terms}'
            print(f'{i}: {factor}{note}')

        choice = []
        for n in range(2):
            selected_cols = input('')
            choice.append(selected_cols)

        factor_a = self.x_cols[int(choice[0])]
        factor_b = self.x_cols[int(choice[1])]

        factor_a_vals = np.linspace(-1, 1, 50)
        factor_b_vals = np.linspace(-1, 1, 50)

        a, b = np.meshgrid(factor_a_vals, factor_b_vals)
        a_flat = a.ravel()
        b_flat = b.ravel()

        base_values = {}
        for factor in x_cols:
            if ' x ' not in factor:
                if factor == factor_a:
                    base_values[factor] = a_flat
                elif factor == factor_b:
                    base_values[factor] = b_flat
                else:
                    print(f'{factor} will be kept constant at 0.')
                    base_values[factor] = np.full_like(a_flat, 0.0)

        raw_cols = {}
        for factor in x_cols:
            if ' x ' in factor:
                parts = factor.split(' x ')
                product = np.ones_like(a_flat)
                for p in parts:
                    product = product * base_values[p]
                raw_cols[factor] = product
            else:
                raw_cols[factor] = base_values[factor]

        raw_matrix = np.column_stack([raw_cols[factor] for factor in x_cols])

        x_poly_grid = poly.transform(raw_matrix)
        y_pred = model.predict(x_poly_grid)
        Z = y_pred.reshape(a.shape)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(projection='3d')
        surf = ax.plot_surface(a, b, Z, cmap='viridis', edgecolor='none', alpha=0.9)
        ax.set_xlabel(factor_a)
        ax.set_ylabel(factor_b)
        ax.set_zlabel(self.y_col)
        ax.set_title(f'Response Surface: {factor_a} x {factor_b}')
        fig.colorbar(surf, shrink=0.5, aspect=10, label=self.y_col)
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(a, b, Z, levels=20, cmap='viridis')
        lines = ax.contour(a, b, Z, levels=10, colors='k', linewidths=0.5)
        ax.clabel(lines, inline=True, fontsize=8)
        ax.set_xlabel(factor_a)
        ax.set_ylabel(factor_b)
        ax.set_title(f'Contour Plot: {factor_a} x {factor_b}')
        fig.colorbar(contour, label=self.y_col)
        plt.tight_layout()
        plt.show()

        max_idx = np.argmax(Z) 
        opt_a = a_flat[max_idx]
        opt_b = b_flat[max_idx]
        opt_response = y_pred[max_idx]

        print(f'Best point found on this grid: {factor_a} = {opt_a:.3f}, {factor_b} = {opt_b:.3f}, '
              f'predicted {self.y_col} = {opt_response:.3f}')

        ax.plot(opt_a, opt_b, 'r*', markersize=15, label='Grid optimum')
        ax.legend()


t = DOE('reaction_DOE.csv')
t.interactions()
t.main_effects()
t.interaction_plots()
t.regression_model(alpha=0.05)
t.RSM()
