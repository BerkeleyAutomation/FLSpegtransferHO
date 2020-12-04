import numpy as np
import FLSpegtransfer.utils.CmnUtil as U


class RANSAC_sphere:
    def fit(self, x, y, z, trials, threshold):
        assert len(x) == len(y) == len(z)

        # find best model
        max_cnt = 0
        ratio = 0
        best_model = []
        for i in range(trials):
            sampled = self.random_sampling(x,y,z)
            model = self.fit_model(sampled)
            cnt = self.eval_model(x, y, z, model, threshold)
            if max_cnt < cnt:
                best_model = model
                max_cnt = cnt
                ratio = max_cnt / len(x)    # inlier / total sample
        return best_model, max_cnt, ratio

    def random_sampling(self, x,y,z):
        # sample three points from data
        sample = []
        for _ in range(4):
            rand = np.random.randint(len(x))
            sample.append([x[rand], y[rand], z[rand]])
        return sample

    def fit_model(self, sample):
        # calculate coefficients from three points
        p1 = sample[0]
        p2 = sample[1]
        p3 = sample[2]
        p4 = sample[3]

        A = np.array([[0, 0, 0, 0, 1],
                      [p1[0]**2+p1[1]**2+p1[2]**2, p1[0], p1[1], p1[2], 1],
                      [p2[0]**2+p2[1]**2+p2[2]**2, p2[0], p2[1], p2[2], 1],
                      [p3[0]**2+p3[1]**2+p3[2]**2, p3[0], p3[1], p3[2], 1],
                      [p4[0]**2+p4[1]**2+p4[2]**2, p4[0], p4[1], p4[2], 1]])

        M11 = np.linalg.det(U.minor(A,0,0))
        if M11 == 0:
            return []
        else:
            M12 = np.linalg.det(U.minor(A,0,1))
            M13 = np.linalg.det(U.minor(A,0,2))
            M14 = np.linalg.det(U.minor(A,0,3))
            M15 = np.linalg.det(U.minor(A,0,4))
            xc = 0.5*M12/M11
            yc = -0.5*M13/M11
            zc = 0.5*M14/M11
            rc = np.sqrt(xc**2 + yc**2 + zc**2 - M15/M11)
            return xc, yc, zc, rc

    def eval_model(self, x,y,z, model, threshold):
        if model == []:
            cnt = 0
        else:
            xc, yc, zc, r = model
            cnt = 0
            for x,y,z in zip(x,y,z):
                dist = np.sqrt((x-xc)**2 + (y-yc)**2 + (z-zc)**2)
                error = abs(dist-r)
                if error < threshold:
                    cnt += 1
        return cnt