import core.my_loader as my_loader
import core.my_filter as my_filter
import core.my_utils as my_utils
from core import my_log_orchestrator



if __name__ == "__main__":
    base_path = 'dataset/'
    df_code_subj = my_loader.load_df_subject(base_path)
    df_code_type = my_loader.load_df_classes(base_path)
    df_code_mov = my_loader.load_df_movements(base_path)

    my_justice = 'TRIBUNAIS_SUPERIORES'
    output_path = 'dataset/stj.xes'

    df, df_subj, df_mov = \
        my_log_orchestrator.load_dataframes('TRIBUNAIS_SUPERIORES',
                                        ['STJ'],
                                        base_path)

    df_mov = my_log_orchestrator.pre_process(
                df_subj,
                df_mov,
                df_code_subj,
                df_code_type,
                df_code_mov
            )

    my_log_orchestrator.create_xes_file(df_mov, output_path)

    print('done!')