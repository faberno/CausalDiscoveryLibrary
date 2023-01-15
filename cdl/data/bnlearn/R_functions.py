from rpy2.robjects.packages import importr
from rpy2 import robjects

bnlearn = importr("bnlearn") # todo: dont make this executed everytime

load = robjects.globalenv.find("load")
set_seed = robjects.globalenv.find("set.seed")
amat = robjects.globalenv.find("amat")
rbn = robjects.globalenv.find("rbn")
as_dataframe = robjects.globalenv.find("as.data.frame")
write_table = robjects.globalenv.find("write.table")
