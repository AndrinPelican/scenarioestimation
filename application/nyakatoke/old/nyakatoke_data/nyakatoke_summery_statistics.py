"""
This script prints descriptive statistics on the Nyakaoke Netork



"""

import numpy as np
from application.nyakatoke.old.nyakatoke_data.nyakatoke_network import di_adj_m

import matplotlib.pyplot as plt

M = di_adj_m

n_edges = sum(sum(M))
n_agents = M.shape[0]
two_stars = M.dot(M)
for i in range(M.shape[0]):
    two_stars[i,i]=0


closed_tryads = np.multiply(M.dot(M),M)
print("agents: "+str(n_agents))
print("edges: "+str(n_edges))
print("reciprocial edges: "+str(sum(sum(np.multiply(M,np.transpose(M,[1,0]))))))



print("two starts "+str(sum(sum(two_stars))))
print("two closed "+str(sum(sum(closed_tryads))))


print("density: "+str(n_edges/(n_agents*(n_agents-1))))
print("transitivity index (->,->,<-): "+str(sum(sum(closed_tryads))/(sum(sum(two_stars)))))

print("max two Star: "+ str(two_stars.max()))
print("max not closed two star: " +str((two_stars-closed_tryads).max()))

plt.imshow(M)
plt.show()
plt.imshow(two_stars)
plt.show()
plt.imshow(closed_tryads)
plt.show()
plt.imshow(two_stars-closed_tryads)
plt.show()


