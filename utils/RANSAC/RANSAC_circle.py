import numpy as np
import FLSpegtransfer.utils.CmnUtil as U

class RANSAC_circle:
    def fit(self, x, y, trials, threshold):
        assert len(x) == len(y)

        # find best model
        max_cnt = 0
        ratio = 0
        best_model = []
        for i in range(trials):
            sampled = self.random_sampling(x,y)
            model = self.fit_model(sampled)
            cnt = self.eval_model(x, y, model, threshold)
            if max_cnt < cnt:
                best_model = model
                max_cnt = cnt
                ratio = max_cnt / len(x)    # inlier / total sample
        return best_model, max_cnt, ratio

    def random_sampling(self, x,y):
        # sample three points from data
        sample = []
        for _ in range(3):
            rand = np.random.randint(len(x))
            sample.append([x[rand], y[rand]])
        return sample

    def fit_model(self, sample):
        # calculate coefficients from three points
        p1 = sample[0]
        p2 = sample[1]
        p3 = sample[2]
        A = np.array([[0, 0, 0, 1],
                      [p1[0]**2+p1[1]**2, p1[0], p1[1], 1],
                      [p2[0]**2+p2[1]**2, p2[0], p2[1], 1],
                      [p3[0]**2+p3[1]**2, p3[0], p3[1], 1]])

        M11 = np.linalg.det(U.minor(A,0,0))
        if M11 == 0:
            return []
        else:
            M12 = np.linalg.det(U.minor(A,0,1))
            M13 = np.linalg.det(U.minor(A,0,2))
            M14 = np.linalg.det(U.minor(A,0,3))
            xc = 0.5*M12/M11
            yc = -0.5*M13/M11
            rc = np.sqrt(xc**2 + yc**2 + M14/M11)
            return xc, yc, rc

    def eval_model(self, x,y, model, threshold):
        if model == []:
            cnt = 0
        else:
            xc, yc, r = model
            cnt = 0
            for x,y in zip(x,y):
                dist = np.sqrt((x - xc)**2 + (y - yc)**2)
                error = abs(dist-r)
                if error < threshold:
                    cnt += 1
        return cnt

# best_model, max_cnt, ratio = self.ransac.fit(x, y, trials=50, threshold=3)