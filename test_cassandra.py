# from cassandra.cluster import Cluster
# import json
# import pandas as pd

# CASSANDRA_IP = '172.17.0.2'
# SOIL_KEYSPACE = 'moreapitest'

# def save_power_index_cql(df, start_date, end_date,
#                            dataset, cp_starts, cp_ends,
#                            weeks_train, query_modelar):

#     cluster = Cluster([CASSANDRA_IP]) # cassandra adress
#     session = cluster.connect(SOIL_KEYSPACE) # soiling keyspace

#     res = session.execute("""CREATE TABLE IF NOT EXISTS power_index_table (
#                              id uuid,
#                              tid int,
#                              timestamp varchar,
#                              pi float,
#                              epl float,
#                              start_date varchar,
#                              end_date varchar,
#                              dataset varchar,
#                              cp_starts varchar,
#                              cp_ends varchar,
#                              weeks_train int,
#                              query_modelar boolean,
#                              PRIMARY KEY ((start_date, end_date, dataset,
#                                            cp_starts, cp_ends, weeks_train,
#                                            query_modelar), id)
#                              )""")

#     res = session.execute("SELECT MAX(tid) from power_index_table")
#     max_tid = res.one().system_max_tid

#     if max_tid is None:
#         next_tid = 0
#     else:
#         next_tid = max_tid + 1

#     stmt = """INSERT INTO power_index_table(id, tid, timestamp, pi, epl,
#                                    start_date, end_date, dataset,
#                                    cp_starts, cp_ends, weeks_train,
#                                    query_modelar)
#                VALUES (uuid(), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

#     prepared = session.prepare(stmt)

#     # save power index to cassandra
#     for index, row in df.iterrows():
#         pi = row.power_index
#         epl = row.estimated_power_lost

#         session.execute(prepared, (next_tid, str(index), pi, epl,
#                                    start_date, end_date, dataset,
#                                    json.dumps(cp_starts), json.dumps(cp_ends),
#                                    weeks_train, query_modelar))


# def load_power_index_cql(start_date, end_date,
#                          dataset, cp_starts, cp_ends,
#                          weeks_train, query_modelar):

#     cluster = Cluster([CASSANDRA_IP]) # cassandra adress
#     session = cluster.connect(SOIL_KEYSPACE) # soiling keyspace

#     def pandas_factory(colnames, rows):
#         return pd.DataFrame(rows, columns=colnames)

#     session.row_factory = pandas_factory
#     session.default_fetch_size = None

#     res = session.execute("""CREATE TABLE IF NOT EXISTS power_index_table (
#                              id uuid,
#                              tid int,
#                              timestamp varchar,
#                              pi float,
#                              epl float,
#                              start_date varchar,
#                              end_date varchar,
#                              dataset varchar,
#                              cp_starts varchar,
#                              cp_ends varchar,
#                              weeks_train int,
#                              query_modelar boolean,
#                              PRIMARY KEY ((start_date, end_date, dataset,
#                                            cp_starts, cp_ends, weeks_train,
#                                            query_modelar), id)
#                              )""") ## added uuid as clustering key

#     res = session.execute("""SELECT timestamp, pi, epl from power_index_table
#                              WHERE start_date=%s AND end_date=%s AND dataset=%s
#                                    AND cp_starts=%s AND cp_ends=%s AND weeks_train=%s
#                                    AND query_modelar=%s """, [start_date, end_date, dataset,
#                                                              json.dumps(cp_starts), json.dumps(cp_ends),
#                                                              weeks_train, query_modelar])
#     res_df = res._current_rows
#     if res_df.empty:
#        return None
#     print(res_df)
#     return res_df


# df = pd.DataFrame([[1, 2], [1, 2]], columns=['power_index', 'estimated_power_lost'])
# df = df.astype(float)
# print(df)

# save_power_index_cql(df, start_date="1-1", end_date='2-2',
#                      dataset='eugene', cp_starts=['asf', 'sdf'], cp_ends=['asdf', 'asdf'],
#                      weeks_train=6, query_modelar=False)

# load_power_index_cql(start_date="1-1", end_date='2-2',
#                            dataset='eugene', cp_starts=['asf', 'sdf'], cp_ends=['asdf', 'asdf'],
#                            weeks_train=6, query_modelar=False)



