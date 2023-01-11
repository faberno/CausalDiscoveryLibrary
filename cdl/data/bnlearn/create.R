library(bnlearn)
arc = arcs(bn)

my_range = 1:

for (i in my_range) {
  i = 1
  data = rbn(bn, 800)
  dadata <- sapply(data, unclass) - 1
  colname = colnames(data)

  len = length(arc) / 2
  nodes = arc[1:len]
  neighbours = arc[53:104]

  nodes = match(nodes, colname) - 1
  neighbours = match(neighbours, colname) - 1


  graph <- data.frame(neighbours, nodes)

  write.csv(data, paste("../INSURANCE/insurance_", i-1, ".csv", sep = ""), row.names = FALSE, col.names=TRUE)
  write.table(graph, paste("../graphs/insurance/graph_", i-1, ".csv", sep = ""), sep=",", row.names = FALSE, col.names=FALSE)
}

# X=t(amat(bn))
# which(X!=0,arr.ind = T)- 1
mydata = data.frame(as.factor(data))
# mydata = mydata[,!names(mydata) %in% "GoodStudent"]
pred = predict(bn, node = "GoodStudent", data = data.frame(mydata))
