import numpy as np
import matplotlib.pyplot as plt

context_lengths = [4, 8, 12, 16, 20]
decoding_random_acc = 0.498146272
decoding_random_std = 0.008550915
encoding_random_acc = 0.501114263
encoding_random_std = 0.013634612

# overlap_decoding_accs = [0.708012561, 0.715594562, 0.704007711, 0.698180894, 0.692479675]
# overlap_decoding_stds = [0.066704047, 0.030023237, 0.051739303, 0.04835408, 0.055407399]
# overlap_encoding_accs = [0.767655997, 0.780813718, 0.760612825, 0.755447154, 0.754735772]
# overlap_encoding_stds = [0.081089999, 0.034330264, 0.066168229, 0.058165803, 0.064537046]

# no_overlap_decoding_accs = [0.708012561, 0.730438312, 0.742597087, 0.731493506, 0.747933468]
# no_overlap_decoding_stds = [0.066704047, 0.055941167, 0.061096987, 0.056711402, 0.0507578]
# no_overlap_encoding_accs = [0.767655997, 0.801684253, 0.813046117, 0.798660714, 0.818699597]
# no_overlap_encoding_stds = [0.081089999, 0.065062929, 0.07552267, 0.076502635, 0.059586057]

# plt.errorbar(context_lengths, overlap_decoding_accs, alpha=.75, fmt=':', yerr=overlap_decoding_stds, label="Overlapping Text Samples", capsize=15, capthick=5, marker='o', markersize=15, linewidth=5)
# plt.fill_between(x=context_lengths, y1=[y - e for y, e in zip(overlap_decoding_accs, overlap_decoding_stds)], y2=[y + e for y, e in zip(overlap_decoding_accs, overlap_decoding_stds)], alpha=.25)
# plt.errorbar(context_lengths, no_overlap_decoding_accs, alpha=.75, fmt=':', yerr=no_overlap_decoding_stds, label="No Overlap", capsize=15, capthick=5, marker='o', markersize=15, linewidth=5)
# plt.fill_between(x=context_lengths, y1=[y - e for y, e in zip(no_overlap_decoding_accs, no_overlap_decoding_stds)], y2=[y + e for y, e in zip(no_overlap_decoding_accs, no_overlap_decoding_stds)], alpha=.25)
# plt.axhline(y = decoding_random_acc+decoding_random_std, color = 'k', linestyle = 'dashed', linewidth=5) 
# plt.axhline(y = decoding_random_acc, color = 'k', label='Randomly Initialized', linewidth=5) 
# plt.axhline(y = decoding_random_acc-decoding_random_std, color = 'k', linestyle = 'dashed', linewidth=5) 
# plt.title('Decoding Accuracy for LOO with Subjects', fontsize=30, weight='bold')
# xi = list(range(len(context_lengths)))
# plt.xticks(context_lengths, context_lengths, fontsize=30)
# plt.yticks(fontsize=30)
# plt.xlabel('Context Length', fontsize=30, weight='bold')
# plt.ylabel('Accuracy', fontsize=30, weight='bold')
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(), fontsize=30)
# plt.show()

# plt.errorbar(context_lengths, overlap_encoding_accs, alpha=.75, fmt=':', yerr=overlap_encoding_stds, label="Overlapping Text Samples", capsize=15, capthick=5, marker='o', markersize=15, linewidth=5)
# plt.fill_between(x=context_lengths, y1=[y - e for y, e in zip(overlap_encoding_accs, overlap_encoding_stds)], y2=[y + e for y, e in zip(overlap_encoding_accs, overlap_encoding_stds)], alpha=.25)
# plt.errorbar(context_lengths, no_overlap_encoding_accs, alpha=.75, fmt=':', yerr=no_overlap_encoding_stds, label="No Overlap", capsize=15, capthick=5, marker='o', markersize=15, linewidth=5)
# plt.fill_between(x=context_lengths, y1=[y - e for y, e in zip(no_overlap_encoding_accs, no_overlap_encoding_stds)], y2=[y + e for y, e in zip(no_overlap_encoding_accs, no_overlap_encoding_stds)], alpha=.25)
# plt.axhline(y = encoding_random_acc+encoding_random_std, color = 'k', linestyle = 'dashed', linewidth=5) 
# plt.axhline(y = encoding_random_acc, color = 'k', label='Randomly Initialized', linewidth=5) 
# plt.axhline(y = encoding_random_acc-encoding_random_std, color = 'k', linestyle = 'dashed', linewidth=5) 
# plt.title('Encoding Accuracy for LOO with Subjects', fontsize=30, weight='bold')
# plt.xticks(context_lengths, context_lengths, fontsize=30)
# plt.yticks(fontsize=30)
# plt.xlabel('Context Length', fontsize=30, weight='bold')
# plt.ylabel('Accuracy', fontsize=30, weight='bold')
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(), fontsize=30)
# plt.show()

runs_overlap_decoding_accs = [0.522195691, 0.514666042, 0.51886782, 0.517504928, 0.519938607]
runs_overlap_decoding_stds = [0.005196212, 0.010977089, 0.018269415, 0.010384454, 0.009369539]
runs_overlap_encoding_accs = [0.531065587, 0.521111171, 0.529738524, 0.526289063, 0.529578711]
runs_overlap_encoding_stds = [0.006895158, 0.016254382, 0.027402721, 0.014358161, 0.016214303]

runs_no_overlap_decoding_accs = [0.522195691, 0.521734683, 0.510370458, 0.514992342, 0.523544085]
runs_no_overlap_decoding_stds = [0.005196212, 0.011287277, 0.010559039, 0.013169008, 0.015831269]
runs_no_overlap_encoding_accs = [0.531065587, 0.528271043, 0.518226039, 0.52592027, 0.537328404]
runs_no_overlap_encoding_stds = [0.006895158, 0.014608995, 0.009828399, 0.012421016, 0.023053506]
plt.title('Decoding Accuracy for Cross-validation with Runs (Within Subject)')

plt.errorbar(context_lengths, runs_overlap_decoding_accs, alpha=.75, fmt=':', yerr=runs_overlap_decoding_stds, label="Overlapping Text Samples", capsize=15, capthick=5, marker='o', markersize=15, linewidth=5)
plt.fill_between(x=context_lengths, y1=[y - e for y, e in zip(runs_overlap_decoding_accs, runs_overlap_decoding_stds)], y2=[y + e for y, e in zip(runs_overlap_decoding_accs, runs_overlap_decoding_stds)], alpha=.25)
plt.errorbar(context_lengths, runs_no_overlap_decoding_accs, alpha=.75, fmt=':', yerr=runs_no_overlap_decoding_stds, label="No Overlap", capsize=15, capthick=5, marker='o', markersize=15, linewidth=5)
plt.fill_between(x=context_lengths, y1=[y - e for y, e in zip(runs_no_overlap_decoding_accs, runs_no_overlap_decoding_stds)], y2=[y + e for y, e in zip(runs_no_overlap_decoding_accs, runs_no_overlap_decoding_stds)], alpha=.25)
plt.axhline(y = decoding_random_acc+decoding_random_std, color = 'k', linestyle = 'dashed', linewidth=5) 
plt.axhline(y = decoding_random_acc, color = 'k', label='Randomly Initialized', linewidth=5) 
plt.axhline(y = decoding_random_acc-decoding_random_std, color = 'k', linestyle = 'dashed', linewidth=5) 
plt.title('Decoding Accuracy for Cross-validation with Runs (Within Subject)', fontsize=30, weight='bold')
xi = list(range(len(context_lengths)))
plt.xticks(context_lengths, context_lengths, fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('Context Length', fontsize=30, weight='bold')
plt.ylabel('Accuracy', fontsize=30, weight='bold')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=30)
plt.show()

plt.errorbar(context_lengths, runs_overlap_encoding_accs, alpha=.75, fmt=':', yerr=runs_overlap_encoding_stds, label="Overlapping Text Samples", capsize=15, capthick=5, marker='o', markersize=15, linewidth=5)
plt.fill_between(x=context_lengths, y1=[y - e for y, e in zip(runs_overlap_encoding_accs, runs_overlap_encoding_stds)], y2=[y + e for y, e in zip(runs_overlap_encoding_accs, runs_overlap_encoding_stds)], alpha=.25)
plt.errorbar(context_lengths, runs_no_overlap_encoding_accs, alpha=.75, fmt=':', yerr=runs_no_overlap_encoding_stds, label="No Overlap", capsize=15, capthick=5, marker='o', markersize=15, linewidth=5)
plt.fill_between(x=context_lengths, y1=[y - e for y, e in zip(runs_no_overlap_encoding_accs, runs_no_overlap_encoding_stds)], y2=[y + e for y, e in zip(runs_no_overlap_encoding_accs, runs_no_overlap_encoding_stds)], alpha=.25)
plt.axhline(y = encoding_random_acc+encoding_random_std, color = 'k', linestyle = 'dashed', linewidth=5) 
plt.axhline(y = encoding_random_acc, color = 'k', label='Randomly Initialized', linewidth=5) 
plt.axhline(y = encoding_random_acc-encoding_random_std, color = 'k', linestyle = 'dashed', linewidth=5) 
plt.title('Encoding Accuracy for Cross-validation with Runs (Within Subject)', fontsize=30, weight='bold')
plt.xticks(context_lengths, context_lengths, fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('Context Length', fontsize=30, weight='bold')
plt.ylabel('Accuracy', fontsize=30, weight='bold')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=30)
plt.show()

# runs_subj_overlap_decoding_accs = [0.637008666, 0.628251609, 0.627024734, 0.637526403, 0.637313766]
# runs_subj_overlap_decoding_stds = [0.017792595, 0.0205699, 0.025568885, 0.020890909, 0.01914418]
# runs_subj_overlap_encoding_accs = [0.67683237, 0.666540748, 0.664712508, 0.678719262, 0.679167246]
# runs_subj_overlap_encoding_stds = [0.023758806, 0.027025127, 0.033768317, 0.024055918, 0.021809054]

# runs_subj_no_overlap_decoding_accs = [0.637008666, 0.66573276, 0.680326834, 0.687751924, 0.685633092]
# runs_subj_no_overlap_decoding_stds = [0.017792595, 0.026586938, 0.027477453, 0.012414692, 0.019100866]
# runs_subj_no_overlap_encoding_accs = [0.67683237, 0.71130761, 0.731228258, 0.740991666, 0.743439732]
# runs_subj_no_overlap_encoding_stds = [0.023758806, 0.0345443, 0.033057344, 0.017994501, 0.028074261]

# plt.errorbar(context_lengths, runs_subj_overlap_decoding_accs, alpha=.75, fmt=':', yerr=runs_subj_overlap_decoding_stds, label="Overlapping Text Samples", capsize=15, capthick=5, marker='o', markersize=15, linewidth=5)
# plt.fill_between(x=context_lengths, y1=[y - e for y, e in zip(runs_subj_overlap_decoding_accs, runs_subj_overlap_decoding_stds)], y2=[y + e for y, e in zip(runs_subj_overlap_decoding_accs, runs_subj_overlap_decoding_stds)], alpha=.25)
# plt.errorbar(context_lengths, runs_subj_no_overlap_decoding_accs, alpha=.75, fmt=':', yerr=runs_subj_no_overlap_decoding_stds, label="No Overlap", capsize=15, capthick=5, marker='o', markersize=15, linewidth=5)
# plt.fill_between(x=context_lengths, y1=[y - e for y, e in zip(runs_subj_no_overlap_decoding_accs, runs_subj_no_overlap_decoding_stds)], y2=[y + e for y, e in zip(runs_subj_no_overlap_decoding_accs, runs_subj_no_overlap_decoding_stds)], alpha=.25)
# plt.axhline(y = decoding_random_acc+decoding_random_std, color = 'k', linestyle = 'dashed', linewidth=5)
# plt.axhline(y = decoding_random_acc, color = 'k', label='Randomly Initialized', linewidth=5) 
# plt.axhline(y = decoding_random_acc-decoding_random_std, color = 'k', linestyle = 'dashed', linewidth=5) 
# plt.title('Decoding Accuracy for Cross-validation with Runs (Across Subjects)', fontsize=30, weight='bold')
# plt.xticks(context_lengths, context_lengths, fontsize=30)
# plt.yticks(fontsize=30)
# plt.xlabel('Context Length', fontsize=30, weight='bold')
# plt.ylabel('Accuracy', fontsize=30, weight='bold')
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(), fontsize=30, loc='lower left')
# plt.show()

# plt.errorbar(context_lengths, runs_subj_overlap_encoding_accs, alpha=.75, fmt=':', yerr=runs_subj_overlap_encoding_stds, label="Overlapping Text Samples", capsize=15, capthick=5, marker='o', markersize=15, linewidth=5)
# plt.fill_between(x=context_lengths, y1=[y - e for y, e in zip(runs_subj_overlap_encoding_accs, runs_subj_overlap_encoding_stds)], y2=[y + e for y, e in zip(runs_subj_overlap_encoding_accs, runs_subj_overlap_encoding_stds)], alpha=.25)
# plt.errorbar(context_lengths, runs_subj_no_overlap_encoding_accs, alpha=.75, fmt=':', yerr=runs_subj_no_overlap_encoding_stds, label="No Overlap", capsize=15, capthick=5, marker='o', markersize=15, linewidth=5)
# plt.fill_between(x=context_lengths, y1=[y - e for y, e in zip(runs_subj_no_overlap_encoding_accs, runs_subj_no_overlap_encoding_stds)], y2=[y + e for y, e in zip(runs_subj_no_overlap_encoding_accs, runs_subj_no_overlap_encoding_stds)], alpha=.25)
# plt.axhline(y = encoding_random_acc+encoding_random_std, color = 'k', linestyle = 'dashed', linewidth=5) 
# plt.axhline(y = encoding_random_acc, color = 'k', label='Randomly Initialized', linewidth=5) 
# plt.axhline(y = encoding_random_acc-encoding_random_std, color = 'k', linestyle = 'dashed', linewidth=5) 
# plt.title('Encoding Accuracy for Cross-validation with Runs (Across Subjects)', fontsize=30, weight='bold')
# plt.xticks(context_lengths, context_lengths, fontsize=30)
# plt.yticks(fontsize=30)
# plt.xlabel('Context Length', fontsize=30, weight='bold')
# plt.ylabel('Accuracy', fontsize=30, weight='bold')
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(), fontsize=30, loc='lower left')
# plt.show()


# epochs_sizes = [5, 10, 50]
# runs_epochs_decoding_accs = [0.522195691, 0.514666042, 0.517504928]
# runs_epochs_decoding_stds = [0.005196212, 0.010977089, 0.010384454]
# runs_epochs_encoding_accs = [0.531065587, 0.521111171, 0.526289063]
# runs_epochs_encoding_stds = [0.006895158, 0.016254382, 0.014358161]

# plt.errorbar(epochs_sizes, runs_epochs_decoding_accs, fmt='bo-', yerr=runs_epochs_decoding_stds, label="Overlapping Text Samples", capsize=10)
# #plt.errorbar(context_lengths, runs_no_overlap_encoding_accs, fmt='ro-', yerr=runs_no_overlap_encoding_stds, label="No Overlap", capsize=10)
# plt.axhline(y = 0.5, color = 'k', linestyle = 'dashed', label='Random') 
# plt.title('Decoding Accuracy for Cross-validation with Runs (Within Subject)')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys())
# plt.show()

# plt.errorbar(epochs_sizes, runs_epochs_encoding_accs, fmt='bo-', yerr=runs_epochs_encoding_stds, label="Overlapping Text Samples", capsize=10)
# #plt.errorbar(context_lengths, runs_no_overlap_encoding_accs, fmt='ro-', yerr=runs_no_overlap_encoding_stds, label="No Overlap", capsize=10)
# plt.axhline(y = 0.5, color = 'k', linestyle = 'dashed', label='Random') 
# plt.title('Encoding Accuracy for Cross-validation with Runs (Within Subject)')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys())
# plt.show()