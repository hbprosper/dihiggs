exec(open('oracle.py').read().split("# Is the excess")[0].replace("print(", "(lambda *a, **k: None)("))
n = T / qt[:, None]; mu_or = f_or / qt[:, None]; mu_en = mu / qt[:, None]
def stats(muhat):
    pos = muhat > 0
    neyman = np.where(T > 0, (n - muhat)**2 / np.where(n > 0, n, np.nan), np.nan)
    pearson = np.where(pos, (n - muhat)**2 / np.where(pos, muhat, np.nan), np.nan)
    dev = np.where(pos, 2 * (muhat - n + np.where(n > 0, n * np.log(np.where(n > 0, n, 1) / np.where(pos, muhat, 1)), 0)), np.nan)
    return neyman, pearson, dev
cnt = np.nanmedian(np.where(T > 0, n, np.nan), 0)
print('median count per bin, first 10 bins:', cnt[:10].round(0), ' bins 10-20:', cnt[10:20].round(0))
for nm, mm in [('free fit', mu_or), ('ensemble', mu_en)]:
    a, b2, c = stats(mm)
    print(f'{nm:9s} Neyman thr {np.nanmean(a[:, :20]):.2f} rest {np.nanmean(a[:, 20:]):.3f} | Pearson thr {np.nanmean(b2[:, :20]):.2f} rest {np.nanmean(b2[:, 20:]):.3f} | deviance thr {np.nanmean(c[:, :20]):.2f} rest {np.nanmean(c[:, 20:]):.3f}')
a, b2, c = stats(mu_or)
print('free fit deviance per bin, first 20:', np.nanmean(c, 0)[:20].round(2))
print('free fit deviance per bin, 20-40:', np.nanmean(c, 0)[20:40].round(2))
