import neat
import sys
import gzip
import pickle

def debug_neat():
    print("Trying to find InnovationTracker...")
    
    # 1. Check module directly
    if hasattr(neat, 'InnovationTracker'):
        print("Found neat.InnovationTracker")
    else:
        print("neat.InnovationTracker NOT found")
        
    # 2. Check DefaultReproduction instance
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "config-feedforward")
    p = neat.Population(config)
    
    print("\nReproduction Attributes:")
    for attr in dir(p.reproduction):
        if 'innovation' in attr:
            print(f" - {attr}: {getattr(p.reproduction, attr)}")
            
    print("\nConfig.GenomeConfig Attributes:")
    for attr in dir(config.genome_config):
        if 'innovation' in attr:
            print(f" - {attr}: {getattr(config.genome_config, attr)}")
            
    # Check what happens when we try to reproduce
    print("\nTrying to reproduce (mock)...")
    try:
        # p.reproduction.reproduce(config, p.species, config.pop_size, p.generation)
        pass
    except Exception as e:
        print(e)

if __name__ == "__main__":
    debug_neat()
