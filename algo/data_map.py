#-------------------------------------------------------------------------------
# @description
#  To ensure the reproducibility of our experiments and avoid the impact of LLM 
#  stochasticity on performance, we preemptively ran the full set of LLM 
#  combinations across all evaluated tasks (cf. Table 3) and repeated five times. 
#  Based on these runs, all algorithms obtain the same average accuracy and cost 
#  of LLM combinations for each corresponding task.
#
#  DataMap aggregates all runs.
#
# @author 
#  Mengsu Ding<dingmengsu.dms@alibaba-inc.com>
#
#-------------------------------------------------------------------------------


from typing import List
import json
import pandas as pd

class DataMap:
    ''' 
        Load data from an input file into a DataFrame object.
        Note that each input file corresponds to a task, with repeated 
        experimental data
    '''
    def __init__(self, input_file: str = None):
        self.data_df = pd.DataFrame(columns=["models", "acc_mean", "acc_std", "acc_min", "acc_max", "prompt_tokens", "completion_tokens"])
        self.model_desc_list: List[str]  = None # model, sort by cost, desc
        self.num_inputs = None
        if input_file is not None:
            self._load(input_file)


    def _load(self, input_file: str):
        """ Load data metrics (e.g., models, accuracy, #tokens) from file.
            The loaded data is ordered by accuracy with descending order. 
        """

        df = pd.read_json(input_file, lines=True)
        grouped = df.groupby(df['models'].apply(lambda x: tuple(x)))
        index = 0
        for name, group in grouped:
            avg_acc = round(group['accuracy'].mean(), 4)
            min_acc = round(group['accuracy'].min(),4)
            max_acc = round(group['accuracy'].max(),4)
            std_acc = round(group['accuracy'].std(),4)
            num_inputs = group['num_inputs'].mean()
            if self.num_inputs is None:
                self.num_inputs = num_inputs
            else:
                assert self.num_inputs == num_inputs
            avg_prompt_tokens = pd.DataFrame(group['prompt_tokens'].tolist()) \
                                .mean().astype(int).values.tolist()
            avg_completion_tokens = pd.DataFrame(group['completion_tokens'] \
                                .tolist()).mean().astype(int).values.tolist()
            self.data_df.loc[index] = [list(name), avg_acc, std_acc, min_acc, max_acc, avg_prompt_tokens, avg_completion_tokens]
            index += 1            
        self._sort_by_acc()

    def _sort_by_acc(self):
        self.data_df = self.data_df.sort_values(by='acc_mean', ascending=False) \
                           .reset_index(drop=True)

    def get_the_best_models_info(self) -> tuple[List[str], float, float]:
        row = self.data_df.loc[0]
        bias = self.data_df['acc_std'].mean()

        # avoid the effect of data skew
        local_bound = 4
        total_sum = 0
        total_count = 0
        for value in self.data_df['acc_std']:
            if value > local_bound:
                continue      
            total_sum += value
            total_count += 1
        filtered_bias = total_sum / total_count
        if bias > filtered_bias:
            bias = filtered_bias
        return row['models'], row['acc_mean'], round(bias, 4)

    def get_accuracy_and_tokens(self, model_list: List[str]) \
        -> tuple[float, List[float], List[float]]:
        row = self.data_df[self.data_df['models'].apply(lambda x: x == model_list)]
        
        return row['acc_mean'].values[0], row['prompt_tokens'].values[0], \
               row['completion_tokens'].values[0]

    def load_partial_map(self, models_list: List[List[str]]):
        partial_map = DataMap()
        partial_map.data_df = self.data_df[self.data_df['models'].isin(models_list)]
        partial_map._sort_by_acc()
        return partial_map

    def dataframe(self):
        return self.data_df

    def get_models_list(self):
        return self.data_df['models'].tolist()

    def get_accuracy(self, model_list: List[str]) -> float:
        row = self.data_df[self.data_df['models'].apply(lambda x: x == model_list)]
        return row['acc_mean'].values[0]

    def print(self):
        pd.set_option('display.max_columns', None)  # None removes column limits
        pd.set_option('display.expand_frame_repr', False)  # Prevents wrapping to new lines
        print(self.data_df)

if __name__ == '__main__':
    single_data_map: DataMap = DataMap("./input/dinsql_bird_all.json")
    for index, row in single_data_map.data_df.iterrows():
        print(f"[{index}] models:{single_data_map.data_df.loc[index]['models']}")
    print(single_data_map.data_df)

    data_map: DataMap = DataMap("./input/dinsql_bird.json")
    print(data_map.data_df)

    model_list: List[str] = ["gpt-4o", "gpt-4o", "gpt-4o"]
    acc, prompt_tokens, completion_tokens = data_map.get_accuracy_and_tokens(model_list)
    print(f"acc: {acc}, prompt_to#kens: {prompt_tokens}, completion_tokens: {completion_tokens}")
