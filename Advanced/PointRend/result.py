from collections import OrderedDict

result = OrderedDict([('sem_seg', {'mIoU': 31.105211120821973, 'fwIoU': 62.61171271106312, 'IoU-wall': 67.53887046856427, 'BoundaryIoU-wall': 71.89026209368424, 'min(IoU, B-Iou)-wall': 67.53887046856427, 'IoU-building': 77.7714075961116, 'BoundaryIoU-building': 9.356590528194744, 'min(IoU, B-Iou)-building': 9.356590528194744, 'IoU-sky': 92.67572080463285, 'BoundaryIoU-sky': 0.0, 'min(IoU, B-Iou)-sky': 0.0, 'IoU-floor': 72.04912198992362, 'BoundaryIoU-floor': 0.0, 'min(IoU, B-Iou)-floor': 0.0, 'IoU-tree': 68.29739828197083, 'BoundaryIoU-tree': 0.0, 'min(IoU, B-Iou)-tree': 0.0, 'IoU-ceiling': 76.38316160786616, 'BoundaryIoU-ceiling': 0.0, 'min(IoU, B-Iou)-ceiling': 0.0, 'IoU-road, route': 76.30563411195035, 'BoundaryIoU-road, route': 0.0, 'min(IoU, B-Iou)-road, route': 0.0, 'IoU-bed': 76.79677408389422, 'BoundaryIoU-bed': 0.0, 'min(IoU, B-Iou)-bed': 0.0, 'IoU-window ': 49.53639213721065, 'BoundaryIoU-window ': 0.0, 'min(IoU, B-Iou)-window ': 0.0, 'IoU-grass': 65.94873886689774, 'BoundaryIoU-grass': 0.0, 'min(IoU, B-Iou)-grass': 0.0, 'IoU-cabinet': 46.75886707210671, 'BoundaryIoU-cabinet': 0.0, 'min(IoU, B-Iou)-cabinet': 0.0, 'IoU-sidewalk, pavement': 51.66071418058983, 'BoundaryIoU-sidewalk, pavement': 0.0, 'min(IoU, B-Iou)-sidewalk, pavement': 0.0, 'IoU-person': 70.33769588055647, 'BoundaryIoU-person': 0.0, 'min(IoU, B-Iou)-person': 0.0, 'IoU-earth, ground': 30.555555555555557, 'BoundaryIoU-earth, ground': 0.0, 'min(IoU, B-Iou)-earth, ground': 0.0, 'IoU-door': 26.051936186149955, 'BoundaryIoU-door': 0.0, 'min(IoU, B-Iou)-door': 0.0, 'IoU-table': 40.13989984926718, 'BoundaryIoU-table': 0.0, 'min(IoU, B-Iou)-table': 0.0, 'IoU-mountain, mount': 51.224185684088106, 'BoundaryIoU-mountain, mount': 0.0, 'min(IoU, B-Iou)-mountain, mount': 0.0, 'IoU-plant': 43.48450676968195, 'BoundaryIoU-plant': 0.0, 'min(IoU, B-Iou)-plant': 0.0, 'IoU-curtain': 57.18442557577783, 'BoundaryIoU-curtain': 0.0, 'min(IoU, B-Iou)-curtain': 0.0, 'IoU-chair': 39.06242397667521, 'BoundaryIoU-chair': 0.0, 'min(IoU, B-Iou)-chair': 0.0, 'IoU-car': 75.06516621876877, 'BoundaryIoU-car': 0.0, 'min(IoU, B-Iou)-car': 0.0, 'IoU-water': 47.85160150662409, 'BoundaryIoU-water': 0.0, 'min(IoU, B-Iou)-water': 0.0, 'IoU-painting, picture': 58.0572934019596, 'BoundaryIoU-painting, picture': 0.0, 'min(IoU, B-Iou)-painting, picture': 0.0, 'IoU-sofa': 48.176412651461575, 'BoundaryIoU-sofa': 0.0, 'min(IoU, B-Iou)-sofa': 0.0, 'IoU-shelf': 32.08558377395389, 'BoundaryIoU-shelf': 0.0, 'min(IoU, B-Iou)-shelf': 0.0, 'IoU-house': 47.491679231971666, 'BoundaryIoU-house': 0.0, 'min(IoU, B-Iou)-house': 0.0, 'IoU-sea': 49.09728333532664, 'BoundaryIoU-sea': 0.0, 'min(IoU, B-Iou)-sea': 0.0, 'IoU-mirror': 37.49667972969136, 'BoundaryIoU-mirror': 0.0, 'min(IoU, B-Iou)-mirror': 0.0, 'IoU-rug': 40.73826169807858, 'BoundaryIoU-rug': 0.0, 'min(IoU, B-Iou)-rug': 0.0, 'IoU-field': 24.20005226986956, 'BoundaryIoU-field': 0.0, 'min(IoU, B-Iou)-field': 0.0, 'IoU-armchair': 27.11477397117759, 'BoundaryIoU-armchair': 0.0, 'min(IoU, B-Iou)-armchair': 0.0, 'IoU-seat': 38.70681661269496, 'BoundaryIoU-seat': 0.0, 'min(IoU, B-Iou)-seat': 0.0, 'IoU-fence': 28.48034183341202, 'BoundaryIoU-fence': 0.0, 'min(IoU, B-Iou)-fence': 0.0, 'IoU-desk': 27.43662099659312, 'BoundaryIoU-desk': 0.0, 'min(IoU, B-Iou)-desk': 0.0, 'IoU-rock, stone': 32.46705169657817, 'BoundaryIoU-rock, stone': 0.0, 'min(IoU, B-Iou)-rock, stone': 0.0, 'IoU-wardrobe, closet, press': 39.45731862593988, 'BoundaryIoU-wardrobe, closet, press': 0.0, 'min(IoU, B-Iou)-wardrobe, closet, press': 0.0, 'IoU-lamp': 43.80362137384878, 'BoundaryIoU-lamp': 0.0, 'min(IoU, B-Iou)-lamp': 0.0, 'IoU-tub': 54.44430320172033, 'BoundaryIoU-tub': 0.0, 'min(IoU, B-Iou)-tub': 0.0, 'IoU-rail': 23.852328566869023, 'BoundaryIoU-rail': 0.0, 'min(IoU, B-Iou)-rail': 0.0, 'IoU-cushion': 35.038356901348024, 'BoundaryIoU-cushion': 0.0, 'min(IoU, B-Iou)-cushion': 0.0, 'IoU-base, pedestal, stand': 7.286875497739811, 'BoundaryIoU-base, pedestal, stand': 0.0, 'min(IoU, B-Iou)-base, pedestal, stand': 0.0, 'IoU-box': 10.542069405894198, 'BoundaryIoU-box': 0.0, 'min(IoU, B-Iou)-box': 0.0, 'IoU-column,  pillar': 30.60741207994912, 'BoundaryIoU-column, pillar': 0.0, 'min(IoU, B-Iou)-column, pillar': 0.0, 'IoU-signboard, sign': 24.322542580461818, 'BoundaryIoU-signboard, sign': 0.0, 'min(IoU, B-Iou)-signboard, sign': 0.0, 'IoU-chest of drawers, chest, bureau, dresser': 36.39253713144875, 'BoundaryIoU-chest of drawers, chest, bureau, dresser': 0.0, 'min(IoU, B-Iou)-chest of drawers, chest, bureau, dresser': 0.0, 'IoU-counter': 22.986658717342433, 'BoundaryIoU-counter': 0.0, 'min(IoU, B-Iou)-counter': 0.0, 'IoU-sand': 30.040843275033215, 'BoundaryIoU-sand': 0.0, 'min(IoU, B-Iou)-sand': 0.0, 'IoU-sink': 47.34111202089913, 'BoundaryIoU-sink': 0.0, 'min(IoU, B-Iou)-sink': 0.0, 'IoU-skyscraper': 45.69381783936283, 'BoundaryIoU-skyscraper': 0.0, 'min(IoU, B-Iou)-skyscraper': 0.0, 'IoU-fireplace': 51.291004140223876, 'BoundaryIoU-fireplace': 0.0, 'min(IoU, B-Iou)-fireplace': 0.0, 'IoU-refrigerator, icebox': 47.326303413926716, 'BoundaryIoU-refrigerator, icebox': 0.0, 'min(IoU, B-Iou)-refrigerator, icebox': 0.0, 'IoU-grandstand, covered stand': 31.32712528518996, 'BoundaryIoU-grandstand, covered stand': 0.0, 'min(IoU, B-Iou)-grandstand, covered stand': 0.0, 'IoU-path': 14.748631401920468, 'BoundaryIoU-path': 0.0, 'min(IoU, B-Iou)-path': 0.0, 'IoU-stairs': 20.188076893875028, 'BoundaryIoU-stairs': 0.0, 'min(IoU, B-Iou)-stairs': 0.0, 'IoU-runway': 64.70038085936255, 'BoundaryIoU-runway': 0.0, 'min(IoU, B-Iou)-runway': 0.0, 'IoU-case, display case, showcase, vitrine': 37.579983515390374, 'BoundaryIoU-case, display case, showcase, vitrine': 0.0, 'min(IoU, B-Iou)-case, display case, showcase, vitrine': 0.0, 'IoU-pool table, billiard table, snooker table': 84.76607178518908, 'BoundaryIoU-pool table, billiard table, snooker table': 0.0, 'min(IoU, B-Iou)-pool table, billiard table, snooker table': 0.0, 'IoU-pillow': 37.20474999937098, 'BoundaryIoU-pillow': 0.0, 'min(IoU, B-Iou)-pillow': 0.0, 'IoU-screen door, screen': 34.9074949186698, 'BoundaryIoU-screen door, screen': 0.0, 'min(IoU, B-Iou)-screen door, screen': 0.0, 'IoU-stairway, staircase': 18.454340628214403, 'BoundaryIoU-stairway, staircase': 0.0, 'min(IoU, B-Iou)-stairway, staircase': 0.0, 'IoU-river': 14.449911691291002, 'BoundaryIoU-river': 0.0, 'min(IoU, B-Iou)-river': 0.0, 'IoU-bridge, span': 24.473328937866942, 'BoundaryIoU-bridge, span': 0.0, 'min(IoU, B-Iou)-bridge, span': 0.0, 'IoU-bookcase': 27.80432138065895, 'BoundaryIoU-bookcase': 0.0, 'min(IoU, B-Iou)-bookcase': 0.0, 'IoU-blind, screen': 11.177277518023098, 'BoundaryIoU-blind, screen': 0.0, 'min(IoU, B-Iou)-blind, screen': 0.0, 'IoU-coffee table': 35.61546316843739, 'BoundaryIoU-coffee table': 0.0, 'min(IoU, B-Iou)-coffee table': 0.0, 'IoU-toilet, can, commode, crapper, pot, potty, stool, throne': 69.9706851689228, 'BoundaryIoU-toilet, can, commode, crapper, pot, potty, stool, throne': 0.0, 'min(IoU, B-Iou)-toilet, can, commode, crapper, pot, potty, stool, throne': 0.0, 'IoU-flower': 24.114849342784684, 'BoundaryIoU-flower': 0.0, 'min(IoU, B-Iou)-flower': 0.0, 'IoU-book': 29.31466930465303, 'BoundaryIoU-book': 0.0, 'min(IoU, B-Iou)-book': 0.0, 'IoU-hill': 6.90318705058224, 'BoundaryIoU-hill': 0.0, 'min(IoU, B-Iou)-hill': 0.0, 'IoU-bench': 30.745245418582616, 'BoundaryIoU-bench': 0.0, 'min(IoU, B-Iou)-bench': 0.0, 'IoU-countertop': 37.304277769521384, 'BoundaryIoU-countertop': 0.0, 'min(IoU, B-Iou)-countertop': 0.0, 'IoU-stove': 51.69286931793422, 'BoundaryIoU-stove': 0.0, 'min(IoU, B-Iou)-stove': 0.0, 'IoU-palm, palm tree': 38.037301973064466, 'BoundaryIoU-palm, palm tree': 0.0, 'min(IoU, B-Iou)-palm, palm tree': 0.0, 'IoU-kitchen island': 30.366887869706748, 'BoundaryIoU-kitchen island': 0.0, 'min(IoU, B-Iou)-kitchen island': 0.0, 'IoU-computer': 42.410709430137025, 'BoundaryIoU-computer': 0.0, 'min(IoU, B-Iou)-computer': 0.0, 'IoU-swivel chair': 29.183128341318447, 'BoundaryIoU-swivel chair': 0.0, 'min(IoU, B-Iou)-swivel chair': 0.0, 'IoU-boat': 44.73040444962139, 'BoundaryIoU-boat': 0.0, 'min(IoU, B-Iou)-boat': 0.0, 'IoU-bar': 21.119729005724164, 'BoundaryIoU-bar': 0.0, 'min(IoU, B-Iou)-bar': 0.0, 'IoU-arcade machine': 21.061023249668956, 'BoundaryIoU-arcade machine': 0.0, 'min(IoU, B-Iou)-arcade machine': 0.0, 'IoU-hovel, hut, hutch, shack, shanty': 10.764202692188492, 'BoundaryIoU-hovel, hut, hutch, shack, shanty': 0.0, 'min(IoU, B-Iou)-hovel, hut, hutch, shack, shanty': 0.0, 'IoU-bus': 69.74542487751879, 'BoundaryIoU-bus': 0.0, 'min(IoU, B-Iou)-bus': 0.0, 'IoU-towel': 30.412753811167477, 'BoundaryIoU-towel': 0.0, 'min(IoU, B-Iou)-towel': 0.0, 'IoU-light': 26.877625291044506, 'BoundaryIoU-light': 0.0, 'min(IoU, B-Iou)-light': 0.0, 'IoU-truck': 15.261499148211243, 'BoundaryIoU-truck': 0.0, 'min(IoU, B-Iou)-truck': 0.0, 'IoU-tower': 18.88740497226217, 'BoundaryIoU-tower': 0.0, 'min(IoU, B-Iou)-tower': 0.0, 'IoU-chandelier': 49.586315415461144, 'BoundaryIoU-chandelier': 0.0, 'min(IoU, B-Iou)-chandelier': 0.0, 'IoU-awning, sunshade, sunblind': 9.70391051302394, 'BoundaryIoU-awning, sunshade, sunblind': 0.0, 'min(IoU, B-Iou)-awning, sunshade, sunblind': 0.0, 'IoU-street lamp': 4.410406759314835, 'BoundaryIoU-street lamp': 0.0, 'min(IoU, B-Iou)-street lamp': 0.0, 'IoU-booth': 9.778128661214238, 'BoundaryIoU-booth': 0.0, 'min(IoU, B-Iou)-booth': 0.0, 'IoU-tv': 46.69794612062269, 'BoundaryIoU-tv': 0.0, 'min(IoU, B-Iou)-tv': 0.0, 'IoU-plane': 42.30129708498801, 'BoundaryIoU-plane': 0.0, 'min(IoU, B-Iou)-plane': 0.0, 'IoU-dirt track': 3.8843994216422875, 'BoundaryIoU-dirt track': 0.0, 'min(IoU, B-Iou)-dirt track': 0.0, 'IoU-clothes': 19.72232729406134, 'BoundaryIoU-clothes': 0.0, 'min(IoU, B-Iou)-clothes': 0.0, 'IoU-pole': 6.7362360561353, 'BoundaryIoU-pole': 0.0, 'min(IoU, B-Iou)-pole': 0.0, 'IoU-land, ground, soil': 7.318738437142103, 'BoundaryIoU-land, ground, soil': 0.0, 'min(IoU, B-Iou)-land, ground, soil': 0.0, 'IoU-bannister, banister, balustrade, balusters, handrail': 1.3265234974201645, 'BoundaryIoU-bannister, banister, balustrade, balusters, handrail': 0.0, 'min(IoU, B-Iou)-bannister, banister, balustrade, balusters, handrail': 0.0, 'IoU-escalator, moving staircase, moving stairway': 10.152863649365223, 'BoundaryIoU-escalator, moving staircase, moving stairway': 0.0, 'min(IoU, B-Iou)-escalator, moving staircase, moving stairway': 0.0, 'IoU-ottoman, pouf, pouffe, puff, hassock': 20.50731475153088, 'BoundaryIoU-ottoman, pouf, pouffe, puff, hassock': 0.0, 'min(IoU, B-Iou)-ottoman, pouf, pouffe, puff, hassock': 0.0, 'IoU-bottle': 9.825800897962974, 'BoundaryIoU-bottle': 0.0, 'min(IoU, B-Iou)-bottle': 0.0, 'IoU-buffet, counter, sideboard': 22.046303866816334, 'BoundaryIoU-buffet, counter, sideboard': 0.0, 'min(IoU, B-Iou)-buffet, counter, sideboard': 0.0, 'IoU-poster, posting, placard, notice, bill, card': 1.0957239529865286, 'BoundaryIoU-poster, posting, placard, notice, bill, card': 0.0, 'min(IoU, B-Iou)-poster, posting, placard, notice, bill, card': 0.0, 'IoU-stage': 0.34581188178580363, 'BoundaryIoU-stage': 0.0, 'min(IoU, B-Iou)-stage': 0.0, 'IoU-van': 16.225501260563675, 'BoundaryIoU-van': 0.0, 'min(IoU, B-Iou)-van': 0.0, 'IoU-ship': 3.8504128158482342, 'BoundaryIoU-ship': 0.0, 'min(IoU, B-Iou)-ship': 0.0, 'IoU-fountain': 11.283600147678293, 'BoundaryIoU-fountain': 0.0, 'min(IoU, B-Iou)-fountain': 0.0, 'IoU-conveyer belt, conveyor belt, conveyer, conveyor, transporter': 37.31327075655461, 'BoundaryIoU-conveyer belt, conveyor belt, conveyer, conveyor, transporter': 0.0, 'min(IoU, B-Iou)-conveyer belt, conveyor belt, conveyer, conveyor, transporter': 0.0, 'IoU-canopy': 3.8854991561729197, 'BoundaryIoU-canopy': 0.0, 'min(IoU, B-Iou)-canopy': 0.0, 'IoU-washer, automatic washer, washing machine': 44.24232646311459, 'BoundaryIoU-washer, automatic washer, washing machine': 0.0, 'min(IoU, B-Iou)-washer, automatic washer, washing machine': 0.0, 'IoU-plaything, toy': 16.113150906361852, 'BoundaryIoU-plaything, toy': 0.0, 'min(IoU, B-Iou)-plaything, toy': 0.0, 'IoU-pool': 20.104094762128284, 'BoundaryIoU-pool': 0.0, 'min(IoU, B-Iou)-pool': 0.0, 'IoU-stool': 15.62515576379444, 'BoundaryIoU-stool': 0.0, 'min(IoU, B-Iou)-stool': 0.0, 'IoU-barrel, cask': 5.855690718213631, 'BoundaryIoU-barrel, cask': 0.0, 'min(IoU, B-Iou)-barrel, cask': 0.0, 'IoU-basket, handbasket': 8.322199715554023, 'BoundaryIoU-basket, handbasket': 0.0, 'min(IoU, B-Iou)-basket, handbasket': 0.0, 'IoU-falls': 40.46129010857788, 'BoundaryIoU-falls': 0.0, 'min(IoU, B-Iou)-falls': 0.0, 'IoU-tent': 68.95253725122434, 'BoundaryIoU-tent': 0.0, 'min(IoU, B-Iou)-tent': 0.0, 'IoU-bag': 3.5748859852830086, 'BoundaryIoU-bag': 0.0, 'min(IoU, B-Iou)-bag': 0.0, 'IoU-minibike, motorbike': 41.17464249721412, 'BoundaryIoU-minibike, motorbike': 0.0, 'min(IoU, B-Iou)-minibike, motorbike': 0.0, 'IoU-cradle': 66.66949501291846, 'BoundaryIoU-cradle': 0.0, 'min(IoU, B-Iou)-cradle': 0.0, 'IoU-oven': 16.136795331287765, 'BoundaryIoU-oven': 0.0, 'min(IoU, B-Iou)-oven': 0.0, 'IoU-ball': 37.05921911468017, 'BoundaryIoU-ball': 0.0, 'min(IoU, B-Iou)-ball': 0.0, 'IoU-food, solid food': 42.89071323706782, 'BoundaryIoU-food, solid food': 0.0, 'min(IoU, B-Iou)-food, solid food': 0.0, 'IoU-step, stair': 3.310650761564344, 'BoundaryIoU-step, stair': 0.0, 'min(IoU, B-Iou)-step, stair': 0.0, 'IoU-tank, storage tank': 17.927860430268648, 'BoundaryIoU-tank, storage tank': 0.0, 'min(IoU, B-Iou)-tank, storage tank': 0.0, 'IoU-trade name': 10.432099268470905, 'BoundaryIoU-trade name': 0.0, 'min(IoU, B-Iou)-trade name': 0.0, 'IoU-microwave': 23.924263145657232, 'BoundaryIoU-microwave': 0.0, 'min(IoU, B-Iou)-microwave': 0.0, 'IoU-pot': 18.612363674496642, 'BoundaryIoU-pot': 0.0, 'min(IoU, B-Iou)-pot': 0.0, 'IoU-animal': 32.563439741805226, 'BoundaryIoU-animal': 0.0, 'min(IoU, B-Iou)-animal': 0.0, 'IoU-bicycle': 42.020731141780374, 'BoundaryIoU-bicycle': 0.0, 'min(IoU, B-Iou)-bicycle': 0.0, 'IoU-lake': 0.08441099043240896, 'BoundaryIoU-lake': 0.0, 'min(IoU, B-Iou)-lake': 0.0, 'IoU-dishwasher': 31.642296276442615, 'BoundaryIoU-dishwasher': 0.0, 'min(IoU, B-Iou)-dishwasher': 0.0, 'IoU-screen': 55.24857620005995, 'BoundaryIoU-screen': 0.0, 'min(IoU, B-Iou)-screen': 0.0, 'IoU-blanket, cover': 1.2978135437825788, 'BoundaryIoU-blanket, cover': 0.0, 'min(IoU, B-Iou)-blanket, cover': 0.0, 'IoU-sculpture': 13.199741569755325, 'BoundaryIoU-sculpture': 0.0, 'min(IoU, B-Iou)-sculpture': 0.0, 'IoU-hood, exhaust hood': 23.88369627353167, 'BoundaryIoU-hood, exhaust hood': 0.0, 'min(IoU, B-Iou)-hood, exhaust hood': 0.0, 'IoU-sconce': 8.19327192014575, 'BoundaryIoU-sconce': 0.0, 'min(IoU, B-Iou)-sconce': 0.0, 'IoU-vase': 11.243199562950991, 'BoundaryIoU-vase': 0.0,  'min(IoU, B-Iou)-vase': 0.0, 'IoU-traffic light': 4.909853788263952, 'BoundaryIoU-traffic light': 0.0, 'min(IoU, B-Iou)-traffic light': 0.0, 'IoU-tray': 0.4030209045190909, 'BoundaryIoU-tray': 0.0, 'min(IoU, B-Iou)-tray': 0.0, 'IoU-trash can': 16.951083753519743, 'BoundaryIoU-trash can': 0.0, 'min(IoU, B-Iou)-trash can': 0.0, 'IoU-fan': 29.452246909622744, 'BoundaryIoU-fan': 0.0, 'min(IoU, B-Iou)-fan': 0.0, 'IoU-pier': 20.2729044834308, 'BoundaryIoU-pier': 0.0, 'min(IoU, B-Iou)-pier': 0.0, 'IoU-crt screen': 7.924274533573422, 'BoundaryIoU-crt screen': 0.0, 'min(IoU, B-Iou)-crt screen': 0.0, 'IoU-plate': 19.445718348851592, 'BoundaryIoU-plate': 0.0, 'min(IoU, B-Iou)-plate': 0.0, 'IoU-monitor': 10.9446629152454, 'BoundaryIoU-monitor': 0.0, 'min(IoU, B-Iou)-monitor': 0.0, 'IoU-bulletin board': 24.208009238076563, 'BoundaryIoU-bulletin board': 0.0, 'min(IoU, B-Iou)-bulletin board': 0.0, 'IoU-shower': 0.0, 'BoundaryIoU-shower': 0.0, 'min(IoU, B-Iou)-shower': 0.0, 'IoU-radiator': 15.02649788362663, 'BoundaryIoU-radiator': 0.0, 'min(IoU, B-Iou)-radiator': 0.0, 'IoU-glass, drinking glass': 0.8119483688615605, 'BoundaryIoU-glass, drinking glass': 0.0, 'min(IoU, B-Iou)-glass, drinking glass': 0.0, 'IoU-clock': 3.867403314917127, 'BoundaryIoU-clock': 0.0, 'min(IoU, B-Iou)-clock': 0.0, 'IoU-flag': 1.8419090662150992, 'BoundaryIoU-flag': 0.0, 'min(IoU, B-Iou)-flag': 0.0, 'mACC': 40.04850187657314, 'pACC': 75.99589899660421, 'ACC-wall': 85.24547384078225, 'ACC-building': 91.47791642838891, 'ACC-sky': 96.80023510817756, 'ACC-floor': 87.78529548002177, 'ACC-tree': 85.67423741848698, 'ACC-ceiling': 86.7586757427282, 'ACC-road, route': 86.68168471897853, 'ACC-bed': 89.09923728849071, 'ACC-window': 69.46515497813994, 'ACC-grass': 80.97015236954813, 'ACC-cabinet': 63.062340952780694, 'ACC-sidewalk, pavement': 70.52490985688708, 'ACC-person': 87.05585194327733, 'ACC-earth, ground': 45.05739619076074, 'ACC-door': 34.41291247259382, 'ACC-table': 58.30160851597803, 'ACC-mountain, mount': 70.11740466677912, 'ACC-plant': 53.69384056449541, 'ACC-curtain': 73.70100212809655, 'ACC-chair': 54.20236138697061, 'ACC-car': 88.4353358107181, 'ACC-water': 65.19337210723725, 'ACC-painting, picture': 72.73326021543642, 'ACC-sofa': 67.81106292090753, 'ACC-shelf': 49.62283327782266, 'ACC-house': 63.2943256958894, 'ACC-sea': 73.50989058802588, 'ACC-mirror': 45.87472054288996, 'ACC-rug': 46.76143618748061, 'ACC-field': 42.448454394226616, 'ACC-armchair': 42.90467593459429, 'ACC-seat': 52.74218753075194, 'ACC-fence': 39.09688636043486, 'ACC-desk': 43.40532686593253, 'ACC-rock, stone': 46.53597930561937, 'ACC-wardrobe, closet, press': 56.85880660145046, 'ACC-lamp': 56.31693015293876, 'ACC-tub': 63.49619704252658, 'ACC-rail': 34.188192990988725, 'ACC-cushion': 45.07768933901639, 'ACC-base, pedestal, stand': 11.554074570391675, 'ACC-box': 15.009233335318998, 'ACC-column, pillar': 39.79243620590328, 'ACC-signboard, sign': 30.731991834354037, 'ACC-chest of drawers, chest, bureau, dresser': 48.792614674692416, 'ACC-counter': 30.690191959417927, 'ACC-sand': 39.4969470303575, 'ACC-sink': 62.37075735451815, 'ACC-skyscraper': 53.77475915769132, 'ACC-fireplace': 61.51191184570436, 'ACC-refrigerator, icebox': 59.16272478714903, 'ACC-grandstand, covered stand': 52.357949011269575, 'ACC-path': 20.694820779290428, 'ACC-stairs': 23.246606452579847, 'ACC-runway': 81.59945260683132, 'ACC-case, display case, showcase, vitrine': 51.817075018680214, 'ACC-pool table, billiard table, snooker table': 92.6298306996623, 'ACC-pillow': 46.474687038703586, 'ACC-screen door, screen': 41.45672963976057, 'ACC-stairway, staircase': 27.370854614025074, 'ACC-river': 23.6843285925793, 'ACC-bridge, span': 32.41832602355649, 'ACC-bookcase': 39.90906439130462, 'ACC-blind, screen': 11.840853516739838, 'ACC-coffee table': 49.622001871521675, 'ACC-toilet, can, commode, crapper, pot, potty, stool, throne': 78.5943341277525, 'ACC-flower': 34.09511246903451, 'ACC-book': 38.19679243321155, 'ACC-hill': 8.59600364079392, 'ACC-bench': 40.566447487482655, 'ACC-countertop': 51.720046935017564, 'ACC-stove': 63.106071545283896, 'ACC-palm, palm tree': 47.752751347323176, 'ACC-kitchen island': 50.48378268622985, 'ACC-computer': 51.075061071851934, 'ACC-swivel chair': 38.93029485791854, 'ACC-boat': 63.6596181075522, 'ACC-bar': 26.484846725499306, 'ACC-arcade machine': 27.240175384770733, 'ACC-hovel, hut, hutch, shack, shanty': 12.296977439736962, 'ACC-bus': 81.56862374617091, 'ACC-towel': 38.453197532495906, 'ACC-light': 29.083428942275376, 'ACC-truck': 20.228055320349984, 'ACC-tower': 23.94917409317328, 'ACC-chandelier': 61.74232947887069, 'ACC-awning, sunshade, sunblind': 11.771533943349997, 'ACC-street lamp': 4.868061642579041, 'ACC-booth': 12.238245440064139, 'ACC-tv': 61.829611466750045, 'ACC-plane': 54.0173707977494, 'ACC-dirt track': 8.424556070720868, 'ACC-clothes': 31.055776044927157, 'ACC-pole': 8.010247462979937, 'ACC-land, ground, soil': 8.601766429555079, 'ACC-bannister, banister, balustrade, balusters, handrail': 1.8641908774680318, 'ACC-escalator, moving staircase, moving stairway': 11.873172708588632, 'ACC-ottoman, pouf, pouffe, puff, hassock': 26.939434254432836, 'ACC-bottle': 10.867452554398497, 'ACC-buffet, counter, sideboard': 24.232369569466904, 'ACC-poster, posting, placard, notice, bill, card': 1.256728181468666, 'ACC-stage': 0.4966590082920999, 'ACC-van': 20.33163791161116, 'ACC-ship': 4.278326122228231,  'ACC-fountain': 13.503751633604203, 'ACC-conveyer belt, conveyor belt, conveyer, conveyor, transporter': 47.31341035736569, 'ACC-canopy': 4.409277603285818, 'ACC-washer, automatic washer, washing machine': 46.59015613585008, 'ACC-plaything, toy': 25.133994179177595, 'ACC-pool': 31.588834513397195, 'ACC-stool': 21.273800752317058, 'ACC-barrel, cask': 39.98179542610081, 'ACC-basket, handbasket': 11.375091583183302, 'ACC-falls': 51.420381360100045, 'ACC-tent': 96.6242915940389, 'ACC-bag': 3.7505944238638436, 'ACC-minibike, motorbike': 50.81501936869559, 'ACC-cradle': 80.65469774889908, 'ACC-oven': 23.69802842501199, 'ACC-ball': 56.1346531805762, 'ACC-food, solid food': 52.022794341329956, 'ACC-step, stair': 3.8347545263765084, 'ACC-tank, storage tank': 18.586716046300342, 'ACC-trade name': 11.232655544376067, 'ACC-microwave': 25.784683019039186, 'ACC-pot': 21.487330661856394, 'ACC-animal': 34.736921542512505, 'ACC-bicycle': 56.094690034004714, 'ACC-lake': 0.08827750892301439, 'ACC-dishwasher': 39.89599043762868, 'ACC-screen': 71.78949789305004, 'ACC-blanket, cover': 1.3913490459221596, 'ACC-sculpture': 20.73871543410992, 'ACC-hood, exhaust hood': 26.220706006039833, 'ACC-sconce': 8.998878310815, 'ACC-vase': 14.973890240482632, 'ACC-traffic light': 5.329290479078229, 'ACC-tray': 0.4818368458541082, 'ACC-trash can': 21.41675215938048, 'ACC-fan': 35.69208730910239, 'ACC-pier': 25.615407094053964, 'ACC-crt screen': 15.316270597993862, 'ACC-plate': 22.40840133234426, 'ACC-monitor': 13.226644961403139, 'ACC-bulletin board': 27.689135053299857, 'ACC-shower': 0.0, 'ACC-radiator': 15.751080738267195, 'ACC-glass, drinking glass': 0.8156858595344858, 'ACC-clock': 4.294519249004361, 'ACC-flag': 1.906796139739977})])


print('mIOU: ', result['sem_seg']['mIoU'])
print('fwIOU: ', result['sem_seg']['fwIoU'])
print('mACC: ', result['sem_seg']['mACC'])
print('pACC: ', result['sem_seg']['pACC'])