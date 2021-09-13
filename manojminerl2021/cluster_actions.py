import joblib
from utils.discretizing import Discretizer
from utils.buffers import DemonstrationBuffer

def run():
    save_dir = '/media/user/997211ec-8c91-4258-b58e-f144225899f4/MinerlV2/dhruvlaad/data/sqil_results'
    
    demonstration_envs = ['MineRLObtainDiamondVectorObf-v0', 'MineRLObtainIronPickaxeVectorObf-v0']
    ddbuffer_file = save_dir + '/ddbuffer.sav'
    DDBuffer = DemonstrationBuffer(envs=demonstration_envs, trim=True, trim_reward=[11,11], load_file=ddbuffer_file)

    action_ddata = DDBuffer.actions

    discretizer = Discretizer(n_actions=60)
    discretizer.cluster_action_data(action_ddata)

    cluster_data = [discretizer.clusterer.labels_, discretizer.n_actions, discretizer.means]

    with open(save_dir + '/cluster_data.sav', 'wb') as f:
        joblib.dump(cluster_data, f)

if __name__ == "__main__":
    run()
        