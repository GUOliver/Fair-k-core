import matplotlib.pyplot as plt

# Amazon graph
x = [5,6,7,8,9,10]
y1 = [220577,165984,112345,81288,64596,49123]
y2 = [200432,158372,96342,41853,21234,10872]

fig, ax = plt.subplots()
ax.plot(x, y1, 'bo-', label='FKC')
ax.plot(x, y2, 'rs--', label='FKC Random')
ax.set_xlabel('k')
ax.set_ylabel('Fair k-Core Size')
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.set_ylim([0, 250000])
ax.set_yticks([0, 50000, 100000, 150000, 200000, 250000])
ax.set_yticklabels(['0', '5', '10$^5$', '15', '20', '25'])
ax.grid(True)
ax.legend()
# add scaling factor text
ax.text(0.02, 1.00, r'$\times 10^5$', transform=ax.transAxes)

plt.savefig('Amazon.png')

# Facebook Company graph
y3 = [3768,3271,2859,2413,1868,1534]
y4 = [3543,2676,1742,1298,978,462]

fig, ax = plt.subplots()
ax.plot(x, y3, 'bo-', label='FKC')
ax.plot(x, y4, 'rs--', label='FKC Random')
ax.set_xlabel('k')
ax.set_ylabel('Fair k-Core Size')
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.set_ylim([0, 4000])
ax.set_yticks([0, 1000, 2000, 3000, 4000])
ax.set_yticklabels(['0', '1', '2', '3', '4'])
ax.grid(True)
ax.legend()

# add scaling factor text
ax.text(0.02, 1.00, r'$\times 10^5$', transform=ax.transAxes)

plt.savefig('Facebook Company.png')
