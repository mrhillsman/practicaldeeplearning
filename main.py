from fastcore.all import *
from fastbook import search_images_ddg
from fastdownload import download_url
from fastai.vision.all import *


def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return search_images_ddg(term, max_images=max_images)


def main():
    urls = search_images('bird photos', max_images=1)
    dest = 'bird.jpg'
    download_url(urls[0], dest, show_progress=False)

    im = Image.open(dest)
    im.thumbnail((256, 256))

    download_url(search_images('forest photos', max_images=1)[0], 'forest.jpg', show_progress=False)
    Image.open('forest.jpg').thumbnail((256, 256))

    searches = 'forest', 'bird'
    path = Path('bird_or_not')
    from time import sleep

    for o in searches:
        dest = (path / o)
        dest.mkdir(exist_ok=True, parents=True)
        download_images(dest, urls=search_images(f'{o} photo', max_images=1))
        download_images(dest, urls=search_images(f'{o} sun photo', max_images=1))
        download_images(dest, urls=search_images(f'{o} shade photo', max_images=1))
        resize_images(path / o, max_size=400, dest=path / o)

    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)
    print(len(failed))

    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')]
    ).dataloaders(path, bs=32)

    dls.show_batch()

    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(3)

    is_bird, _, probs = learn.predict(PILImage.create('bird.jpg'))
    print(f"This is a: {is_bird}.")
    print(f"Probability it's a bird: {probs[0]:.4f}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
